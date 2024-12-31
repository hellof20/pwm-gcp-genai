package genai

import (
	"encoding/base64"
	"os"
	"testing"
)

func TestGeminiInvokeText(t *testing.T) {
	gemini := NewGeminiAPI("us-central1", "speedy-victory-336109", "gemini-1.5-flash-002", 1, 3, 1)
	resp, err := gemini.InvokeText("who are you?")
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}

func TestGeminiInvokeImage(t *testing.T) {
	gemini := NewGeminiAPI("us-central1", "speedy-victory-336109", "gemini-1.5-flash-002", 1, 3, 1)
	image_paths := []string{"test1.jpeg"}
	resp, err := gemini.InvokeImages("描述图片内容，输出语言为中文", image_paths)
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}

func TestClaudeInvokeText(t *testing.T) {
	claude := NewClaudeAPI("us-east5", "speedy-victory-336109", "claude-3-5-sonnet@20240620", 1)
	resp, err := claude.InvokeMessages([]*Message{
		{
			Role: "user",
			Contents: []interface{}{
				&MessageText{
					Type: "text",
					Text: "who are you?",
				},
			},
		},
	})
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}

func TestClaudeInvokeImage(t *testing.T) {
	claude := NewClaudeAPI("us-east5", "speedy-victory-336109", "claude-3-5-sonnet@20240620", 1)
	bytes, err := os.ReadFile("test1.jpeg")
	if err != nil {
		t.Error(err)
	}
	messages := []*Message{
		{
			Role: "user",
			Contents: []interface{}{
				&MessageText{
					Type: "text",
					Text: "描述这个图片内容，输出语言为中文",
				},
				&MessageImage{
					Type: "image",
					Source: ImageSource{
						Type:      "base64",
						MediaType: "image/jpeg",
						Data:      base64.StdEncoding.EncodeToString(bytes),
					},
				},
			},
		},
	}

	resp, err := claude.InvokeMessages(messages)
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}
