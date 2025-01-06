package genai

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
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
func NewClaudeAPI(location, projectID, model string) *ClaudeAPI {
	api := &ClaudeAPI{
		Location:  location,
		Model:     model,
		ProjectID: projectID,
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

// CluadeInvokeMessages 调用 Claude API
func (a *ClaudeAPI) RawPredict(requestBody string) ([]byte, error) {
	token, err := a.getAccessToken()
	if err != nil {
		return nil, fmt.Errorf("failed to get access token: %w", err)
	}

	apiURL := fmt.Sprintf(apiURLFormat, a.Location, a.ProjectID, a.Location, a.Model)
	log.Println(apiURL)
	bodyBuffer := bytes.NewBufferString(requestBody)
	request, err := http.NewRequestWithContext(context.Background(), http.MethodPost, apiURL, bodyBuffer)
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

	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d, response body: %s", response.StatusCode, string(responseBody))
	}

	return responseBody, nil
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
