package genai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

type ClaudeAPI struct {
	location  string
	projectID string
	model     string
	temp      float32
	token     *oauth2.Token
	tokenMtx  sync.Mutex
}

// NewAPI 创建 API 实例
func NewClaudeAPI(location, projectID, model string, temp float32) *ClaudeAPI {
	return &ClaudeAPI{
		location:  location,
		model:     model,
		projectID: projectID,
		temp:      temp,
	}
}

// Message 定义消息结构
type Message struct {
	Role     string        `json:"role"`
	Contents []interface{} `json:"contents"`
}

// MessageText 定义文本消息结构
type MessageText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type MessageImage struct {
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
	Temperature      float64                  `json:"temperature"`
	MaxTokens        int                      `json:"max_tokens"`
	TopP             float64                  `json:"top_p"`
	TopK             int                      `json:"top_k"`
}

// ClaudeResponse 定义响应结构
type ClaudeResponse struct {
	Content []interface{} `json:"content"`
}

// CluadeInvokeMessages 调用 Claude API
func (a *ClaudeAPI) InvokeMessages(messages []*Message) (string, error) {

	request := ClaudeRequest{
		AnthropicVersion: "vertex-2023-10-16",
		Temperature:      float64(a.temp),
		MaxTokens:        1024,
		TopP:             0.95,
		TopK:             40,
	}

	for _, msg := range messages {
		if msg.Role == "system" {
			for _, content := range msg.Contents {
				switch v := content.(type) {
				case *MessageText:
					request.System = v.Text
				}
			}
			continue
		}

		request.Messages = append(request.Messages, map[string]interface{}{
			"role":    msg.Role,
			"content": msg.Contents,
		})
	}

	payloadBytes, err := json.Marshal(request)
	if err != nil {
		return "", err
	}

	output, err := a.callClaudeModel(payloadBytes)
	if err != nil {
		return "", err
	}

	var resp ClaudeResponse
	err = json.Unmarshal(output, &resp)
	if err != nil {
		return "", err
	}

	var contents []string
	for _, v := range resp.Content {
		switch vv := v.(type) {
		case map[string]interface{}:
			if vv["type"] == "text" {
				contents = append(contents, vv["text"].(string))
			}
		}
	}

	return strings.Join(contents, "\n"), nil
}

// callClaudeModel 调用 Claude 模型
func (a *ClaudeAPI) callClaudeModel(payloadBytes []byte) ([]byte, error) {
	token, err := a.getAccessToken()
	if err != nil {
		return nil, err
	}

	apiURL := fmt.Sprintf("https://%v-aiplatform.googleapis.com/v1/projects/%v/locations/%v/publishers/anthropic/models/%v:rawPredict", a.location, a.projectID, a.location, a.model)

	request, err := http.NewRequestWithContext(context.Background(), "POST", apiURL, strings.NewReader(string(payloadBytes)))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set("Content-Type", "application/json; charset=utf-8")

	client := http.Client{}
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}

	defer response.Body.Close()

	if response.StatusCode == http.StatusOK {
		body, err := io.ReadAll(response.Body)
		if err != nil {
			return nil, err
		}

		return body, nil
	}

	return nil, fmt.Errorf("unexpected status code: %d", response.StatusCode)
}

// getAccessToken 获取 Google Cloud OAuth token
func (a *ClaudeAPI) getAccessToken() (string, error) {
	a.tokenMtx.Lock()
	defer a.tokenMtx.Unlock()

	// 如果令牌存在且未过期，则直接返回
	if a.token != nil && !a.token.Expiry.Before(time.Now()) {
		return a.token.AccessToken, nil
	}

	creds, err := google.FindDefaultCredentials(context.Background(), "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return "", err
	}

	token, err := creds.TokenSource.Token()
	if err != nil {
		return "", err
	}

	a.token = token

	return token.AccessToken, nil
}
