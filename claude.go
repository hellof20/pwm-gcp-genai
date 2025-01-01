package genai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

const (
	anthropicVersion = "vertex-2023-10-16"
	contentType      = "application/json; charset=utf-8"
	apiURLFormat     = "https://%v-aiplatform.googleapis.com/v1/projects/%v/locations/%v/publishers/anthropic/models/%v:rawPredict"
)

type Contents interface{}

type Message struct {
	Role     string     `json:"role"`
	Contents []Contents `json:"contents"`
}

// MessageText 定义文本消息结构
type ContentText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ContentImage struct {
	Type   string      `json:"type"`
	Source ImageSource `json:"source"`
}

type ImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

// ClaudeRequest 定义请求结构
type ClaudeRequest struct {
	AnthropicVersion string                   `json:"anthropic_version"`
	Messages         []map[string]interface{} `json:"messages"`
	System           string                   `json:"system,omitempty"`
	Temperature      float32                  `json:"temperature"`
	MaxTokens        int                      `json:"max_tokens"`
	TopP             float32                  `json:"top_p"`
	TopK             int                      `json:"top_k"`
}

// ClaudeResponse 定义响应结构
type ClaudeResponse struct {
	Content []map[string]interface{} `json:"content"`
}

type ClaudeAPI struct {
	Location    string
	ProjectID   string
	Model       string
	Temperature float32
	Token       *oauth2.Token
	TokenMtx    sync.Mutex
	MaxRetries  int
	RetryDelay  time.Duration
	httpClient  *http.Client // 复用 http client
}

// NewAPI 创建 API 实例
func NewClaudeAPI(location, projectID, model string, temp float32, maxRetries int, retryDelay time.Duration) *ClaudeAPI {
	api := &ClaudeAPI{
		Location:    location,
		Model:       model,
		ProjectID:   projectID,
		Temperature: temp,
		MaxRetries:  maxRetries,
		RetryDelay:  retryDelay,
	}
	api.initHttpClient()
	return api
}

func (a *ClaudeAPI) initHttpClient() {
	if a.httpClient == nil {
		a.httpClient = &http.Client{
			Timeout: 60 * time.Second, // 设置超时时间
		}
	}
}

func (a *ClaudeAPI) buildMessages(prompts []string, img_paths []string) ([]*Message, error) {
	var messages []*Message
	var contents []Contents
	for _, prompt := range prompts {
		contents = append(contents, &ContentText{
			Type: "text",
			Text: prompt,
		})
	}

	for _, img_path := range img_paths {
		bytes, err := os.ReadFile(img_path)
		if err != nil {
			return nil, fmt.Errorf("failed to read image file %s: %w", img_path, err)
		}
		contents = append(contents, &ContentImage{
			Type: "image",
			Source: ImageSource{
				Type:      "base64",
				MediaType: "image/jpeg",
				Data:      base64.StdEncoding.EncodeToString(bytes),
			},
		})
	}

	messages = append(messages, &Message{
		Role:     "user",
		Contents: contents,
	})
	return messages, nil
}

func (a *ClaudeAPI) Invoke(prompts []string, img_paths []string) (string, error) {
	messages, err := a.buildMessages(prompts, img_paths)
	if err != nil {
		return "", err
	}

	resp, err := a.invokeMessages(messages)
	if err != nil {
		return "", err
	}

	return resp, nil
}

// CluadeInvokeMessages 调用 Claude API
func (a *ClaudeAPI) invokeMessages(messages []*Message) (string, error) {
	request, err := a.buildClaudeRequest(messages)
	if err != nil {
		return "", fmt.Errorf("failed to build request: %w", err)
	}

	payloadBytes, err := json.Marshal(request)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	output, err := a.callClaudeModel(payloadBytes)
	if err != nil {
		return "", fmt.Errorf("failed to call claude model: %w", err)
	}

	var resp ClaudeResponse
	if err := json.Unmarshal(output, &resp); err != nil {
		return "", fmt.Errorf("failed to unmarshal response: %w", err)
	}

	contentStr, err := a.parseResponseContent(resp)

	if err != nil {
		return "", fmt.Errorf("failed to parse response content: %w", err)
	}

	return contentStr, nil
}

func (a *ClaudeAPI) buildClaudeRequest(messages []*Message) (ClaudeRequest, error) {
	request := ClaudeRequest{
		AnthropicVersion: anthropicVersion,
		Temperature:      a.Temperature,
		MaxTokens:        1024,
		TopP:             0.95,
		TopK:             40,
	}

	for _, msg := range messages {
		if msg.Role == "system" {
			systemText, err := a.extractSystemText(msg.Contents)
			if err != nil {
				return ClaudeRequest{}, err
			}
			request.System = systemText
			continue
		}

		request.Messages = append(request.Messages, map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Contents,
		})
	}
	return request, nil
}

func (a *ClaudeAPI) extractSystemText(contents []Contents) (string, error) {
	for _, content := range contents {
		if textContent, ok := content.(*ContentText); ok {
			return textContent.Text, nil
		}
	}
	return "", fmt.Errorf("system message must contain text content")
}

func (a *ClaudeAPI) parseResponseContent(resp ClaudeResponse) (string, error) {
	var contents []string
	for _, v := range resp.Content {
		if v["type"] == "text" {
			text, ok := v["text"].(string)
			if !ok {
				return "", errors.New("invalid text content format")
			}
			contents = append(contents, text)
		}
	}

	return strings.Join(contents, "\n"), nil
}

// callClaudeModel 调用 Claude 模型
func (a *ClaudeAPI) callClaudeModel(payloadBytes []byte) ([]byte, error) {
	var body []byte
	var err error
	for i := 0; i <= a.MaxRetries; i++ {
		body, err = a.callClaudeModelInternal(payloadBytes)
		if err == nil {
			return body, nil
		}
		if i == a.MaxRetries {
			return nil, fmt.Errorf("failed after %d retries: %w", a.MaxRetries, err)
		}
		time.Sleep(a.RetryDelay * time.Duration(math.Pow(2, float64(i)))) // 指数退避
	}
	return nil, errors.New("unreachable")

}

func (a *ClaudeAPI) callClaudeModelInternal(payloadBytes []byte) ([]byte, error) {
	token, err := a.getAccessToken()
	if err != nil {
		return nil, fmt.Errorf("failed to get access token: %w", err)
	}

	apiURL := fmt.Sprintf(apiURLFormat, a.Location, a.ProjectID, a.Location, a.Model)

	request, err := http.NewRequestWithContext(context.Background(), http.MethodPost, apiURL, strings.NewReader(string(payloadBytes)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set("Content-Type", contentType)

	response, err := a.httpClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	defer response.Body.Close()

	body, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d, response body: %s", response.StatusCode, string(body))
	}

	return body, nil
}

// getAccessToken 获取 Google Cloud OAuth token
func (a *ClaudeAPI) getAccessToken() (string, error) {
	a.TokenMtx.Lock()
	defer a.TokenMtx.Unlock()

	// 如果令牌存在且未过期，则直接返回
	if a.Token != nil && !a.Token.Expiry.Before(time.Now()) {
		return a.Token.AccessToken, nil
	}

	creds, err := google.FindDefaultCredentials(context.Background(), "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return "", err
	}

	token, err := creds.TokenSource.Token()
	if err != nil {
		return "", err
	}

	a.Token = token

	return token.AccessToken, nil
}
