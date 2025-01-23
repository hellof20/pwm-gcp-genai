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

	"cloud.google.com/go/storage"
	"cloud.google.com/go/vertexai/genai"
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
			fmt.Printf("Retrying after %v, attempt %d/%d, error: %v\n", delay, retry+1, a.MaxRetries, err)
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
		ctx := context.Background()
		mimeType, err := getGCSFileMimeTypeFromMetadata(ctx, b.Path)
		if err != nil {
			return nil, fmt.Errorf("failed to get GCS file mime type: %w", err)
		}
		// 如果是 GCS 路径，使用 genai.FileData
		return genai.FileData{
			MIMEType: mimeType,
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
		mimeType := http.DetectContentType(data)
		if mimeType == "application/octet-stream" { // fallback to extension
			mimeType = mime.TypeByExtension(filepath.Ext(b.Path))
		}
		return genai.Blob{
			MIMEType: mimeType,
			Data:     data,
		}, nil
	} else {
		// 如果是本地路径，使用 genai.Blob
		data, err := os.ReadFile(b.Path)
		if err != nil {
			return nil, fmt.Errorf("failed to read file: %w", err)
		}
		mimeType := http.DetectContentType(data)
		if mimeType == "application/octet-stream" { // fallback to extension
			mimeType = mime.TypeByExtension(filepath.Ext(b.Path))
		}
		return genai.Blob{
			MIMEType: mimeType,
			Data:     data,
		}, nil
	}
}

func (a *GeminiAPI) Invoke(inputs ...Input) (string, error) {
	ctx, cancelFn := context.WithTimeout(context.Background(), 180*time.Second)
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

func downloadFile(urlStr string) (string, error) {
	parsedURL, err := url.Parse(urlStr)
	if err != nil {
		return "", fmt.Errorf("invalid URL: %w", err)
	}
	resp, err := http.Get(urlStr)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("failed to download file, status code: %d", resp.StatusCode)
	}

	tmpFile, err := os.CreateTemp("", filepath.Base(parsedURL.Path)) // 使用原始文件名创建临时文件
	if err != nil {
		return "", err
	}
	defer tmpFile.Close()

	_, err = io.Copy(tmpFile, resp.Body)
	if err != nil {
		os.Remove(tmpFile.Name())
		return "", err
	}
	return tmpFile.Name(), nil
}

func getGCSFileMimeTypeFromMetadata(ctx context.Context, gcsPath string) (string, error) {
	// Parse the GCS path
	u, err := url.Parse(gcsPath)
	if err != nil {
		return "", fmt.Errorf("invalid GCS path: %w", err)
	}
	if u.Scheme != "gs" {
		return "", fmt.Errorf("invalid GCS scheme: %s", u.Scheme)
	}
	bucketName := u.Host
	objectName := strings.TrimPrefix(u.Path, "/")

	client, err := storage.NewClient(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to create GCS client: %w", err)
	}
	defer client.Close()

	obj := client.Bucket(bucketName).Object(objectName)
	attrs, err := obj.Attrs(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to get GCS object attributes: %w", err)
	}
	mimeType := attrs.ContentType
	if mimeType == "" { // fallback to extension
		mimeType = mime.TypeByExtension(filepath.Ext(objectName))
	}
	return mimeType, nil
}
