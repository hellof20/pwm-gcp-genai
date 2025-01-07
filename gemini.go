package genai

import (
	"context"
	"fmt"
	"io"
	"math"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"cloud.google.com/go/vertexai/genai"
	"github.com/google/uuid"
)

type GeminiAPI struct {
	ModelName        string
	ProjectID        string
	Location         string
	ResponseSchema   *genai.Schema
	ResponseMIMEType string
	Client           *genai.Client
	Temperature      float32
	MaxRetries       int           // 最大重试次数
	RetryDelay       time.Duration // 初始重试延迟
}

// NewAPI 创建 API 实例
func NewGeminiAPI(location, projectID, model string, temperature float32, maxRetries int, retryDelay time.Duration) *GeminiAPI {
	return &GeminiAPI{
		Location:    location,
		ModelName:   model,
		ProjectID:   projectID,
		Temperature: temperature,
		MaxRetries:  maxRetries,
		RetryDelay:  retryDelay,
	}
}

// 初始化客户端，避免每次调用都创建
func (a *GeminiAPI) InitClient(ctx context.Context) error {
	if a.Client != nil {
		return nil // 已初始化，直接返回
	}
	client, err := genai.NewClient(ctx, a.ProjectID, a.Location)
	if err != nil {
		return fmt.Errorf("failed to create client: %w", err)
	}
	a.Client = client
	return nil
}

// 重试函数
func (a *GeminiAPI) retryableGenerateContent(ctx context.Context, model *genai.GenerativeModel, parts ...genai.Part) (*genai.GenerateContentResponse, error) {
	var resp *genai.GenerateContentResponse
	var err error
	for retry := 0; retry <= a.MaxRetries; retry++ {
		resp, err = model.GenerateContent(ctx, parts...)
		if err == nil {
			break
		}
		if retry < a.MaxRetries {
			delay := a.RetryDelay * time.Duration(math.Pow(2, float64(retry))) // 指数退避
			fmt.Printf("Retrying after %v, attempt %d/%d, error: %v\n", delay, retry+1, a.MaxRetries+1, err)
			time.Sleep(delay)
			continue
		} else {
			// 达到最大重试次数，返回错误
			return nil, fmt.Errorf("max retries reached after %d attempts, last error: %w", a.MaxRetries, err)
		}
	}
	return resp, nil
}

// 定义一个接口，用于表示各种输入类型
type Input interface {
	ToPart() (genai.Part, error)
}

// 实现文本输入
type TextInput struct {
	Text string
}

func (t TextInput) ToPart() (genai.Part, error) {
	return genai.Text(t.Text), nil
}

// 实现其他模态输入
type BlobInput struct {
	Path string
}

func (b BlobInput) ToPart() (genai.Part, error) {
	if strings.HasPrefix(b.Path, "gs://") {
		// 如果是 GCS 路径，使用 genai.FileData
		return genai.FileData{
			MIMEType: mime.TypeByExtension(filepath.Ext(b.Path)),
			FileURI:  b.Path,
		}, nil
	} else if strings.HasPrefix(b.Path, "http://") || strings.HasPrefix(b.Path, "https://") {
		// 如果是 HTTP/HTTPS 路径，下载文件并转换为 Blob
		tmpFile, err := downloadFile(b.Path)
		if err != nil {
			return nil, fmt.Errorf("failed to download file: %w", err)
		}
		defer os.Remove(tmpFile) // 确保函数退出时删除临时文件
		data, err := os.ReadFile(tmpFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read downloaded file: %w", err)
		}
		return genai.Blob{
			MIMEType: mime.TypeByExtension(filepath.Ext(b.Path)),
			Data:     data,
		}, nil
	} else {
		// 如果是本地路径，使用 genai.Blob
		data, err := os.ReadFile(b.Path)
		if err != nil {
			return nil, fmt.Errorf("failed to read file: %w", err)
		}
		return genai.Blob{
			MIMEType: mime.TypeByExtension(filepath.Ext(b.Path)),
			Data:     data,
		}, nil
	}
}

func (a *GeminiAPI) Invoke(inputs ...Input) (string, error) {
	ctx, cancelFn := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancelFn()

	// 初始化客户端
	err := a.InitClient(ctx)
	if err != nil {
		return "", err
	}

	client := a.Client.GenerativeModel(a.ModelName)
	client.SetTemperature(a.Temperature)
	client.GenerationConfig.ResponseMIMEType = a.ResponseMIMEType
	client.GenerationConfig.ResponseSchema = a.ResponseSchema
	client.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: genai.HarmBlockNone,
		},
		{
			Category:  genai.HarmCategoryHateSpeech,
			Threshold: genai.HarmBlockNone,
		},
		{
			Category:  genai.HarmCategorySexuallyExplicit,
			Threshold: genai.HarmBlockNone,
		},
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: genai.HarmBlockNone,
		},
	}

	var parts []genai.Part
	// 处理可变参数 inputs
	for _, input := range inputs {
		part, err := input.ToPart()
		if err != nil {
			return "", err
		}
		parts = append(parts, part)
	}

	resp, err := a.retryableGenerateContent(ctx, client, parts...)
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %w", err)
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return "", fmt.Errorf("no response content found")
	}

	result := resp.Candidates[0].Content.Parts[0]
	resultStr := fmt.Sprint(result)
	return resultStr, nil
}

// downloadFile downloads a file from a given URL and returns the local path
func downloadFile(urlStr string) (string, error) {
	parsedURL, err := url.Parse(urlStr)
	if err != nil {
		return "", fmt.Errorf("failed to parse URL: %w", err)
	}
	// 获取URL路径部分，并从中提取出文件名
	urlPath := parsedURL.Path
	fileExtension := filepath.Ext(urlPath)
	if fileExtension == "" {
		fileExtension = ".tmp"
	}
	fileName := uuid.New().String() + fileExtension

	// 创建临时文件
	tmpFile, err := os.CreateTemp("", fileName)
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer tmpFile.Close()

	resp, err := http.Get(urlStr)
	if err != nil {
		os.Remove(tmpFile.Name())
		return "", fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		os.Remove(tmpFile.Name())
		return "", fmt.Errorf("failed to download file: status code %d", resp.StatusCode)
	}

	_, err = io.Copy(tmpFile, resp.Body)
	if err != nil {
		os.Remove(tmpFile.Name())
		return "", fmt.Errorf("failed to write to temp file: %w", err)
	}

	return tmpFile.Name(), nil
}
