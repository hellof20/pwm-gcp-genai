package genai

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

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

func (a *GeminiAPI) Invoke(prompts []string, img_paths []string) (string, error) {
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

	var parts []genai.Part // 使用 parts 切片来存储所有的输入

	// 添加文本 prompt
	for _, prompt := range prompts {
		parts = append(parts, genai.Text(prompt))
	}

	// 添加图片 prompt
	for _, img_path := range img_paths {
		bytes, err := os.ReadFile(img_path)
		if err != nil {
			return "", fmt.Errorf("failed to read image file %s: %w", img_path, err)
		}

		// 根据文件扩展名推断MIME类型
		mimeType := "image/jpeg" // 默认 JPEG
		ext := filepath.Ext(img_path)
		if ext == ".png" {
			mimeType = "image/png"
		} else if ext == ".webp" {
			mimeType = "image/webp"
		}
		img_data := genai.ImageData(mimeType, bytes)
		parts = append(parts, img_data) // 添加图片数据到 parts 切片
	}

	resp, err := a.retryableGenerateContent(ctx, client, parts...) // 使用 ... 展开 parts 切片
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
