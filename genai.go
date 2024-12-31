package genai

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
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

// 判断错误是否需要重试
func shouldRetry(err error) bool {
	if err == nil {
		return false
	}

	if strings.Contains(err.Error(), "429") ||
		strings.Contains(err.Error(), "529") {
		return true
	}
	return false
}

// 抽象的重试函数
func (a *GeminiAPI) retryableGenerateContent(ctx context.Context, model *genai.GenerativeModel, parts ...genai.Part) (*genai.GenerateContentResponse, error) {
	var resp *genai.GenerateContentResponse
	var err error

	for retry := 0; retry <= a.MaxRetries; retry++ {
		resp, err = model.GenerateContent(ctx, parts...)
		if err == nil {
			// 成功, 退出重试循环
			break
		}

		// 检查是否为需要重试的错误
		if shouldRetry(err) {
			if retry < a.MaxRetries {
				delay := a.RetryDelay * time.Duration(math.Pow(2, float64(retry))) // 指数退避
				fmt.Printf("Retrying after %v, attempt %d/%d, error: %v\n", delay, retry+1, a.MaxRetries+1, err)
				time.Sleep(delay)
				continue
			} else {
				// 达到最大重试次数，返回错误
				return nil, fmt.Errorf("max retries reached after %d attempts, last error: %w", a.MaxRetries, err)
			}

		} else {
			// 不需要重试的错误，直接返回错误
			return nil, fmt.Errorf("generate content failed: %w", err)
		}

	}

	return resp, nil
}

func (a *GeminiAPI) InvokeText(prompt string) (string, error) {
	ctx := context.Background()

	// 初始化客户端
	err := a.InitClient(ctx)
	if err != nil {
		return "", err
	}

	client := a.Client.GenerativeModel(a.ModelName)
	client.SetTemperature(a.Temperature)
	client.GenerationConfig.ResponseMIMEType = a.ResponseMIMEType
	client.GenerationConfig.ResponseSchema = a.ResponseSchema

	resp, err := a.retryableGenerateContent(ctx, client, genai.Text(prompt))
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

func (a *GeminiAPI) InvokeImage(prompt string, img_paths []string) (string, error) {
	ctx := context.Background()

	// 初始化客户端
	err := a.InitClient(ctx)
	if err != nil {
		return "", err
	}

	model := a.Client.GenerativeModel(a.ModelName)
	model.SetTemperature(a.Temperature)
	model.GenerationConfig.ResponseMIMEType = a.ResponseMIMEType
	model.GenerationConfig.ResponseSchema = a.ResponseSchema

	var parts []genai.Part // 使用 parts 切片来存储所有的输入

	// 添加文本 prompt
	parts = append(parts, genai.Text(prompt))

	// 遍历图片路径列表
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

	resp, err := model.GenerateContent(ctx, parts...) // 使用 ... 展开 parts 切片
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
