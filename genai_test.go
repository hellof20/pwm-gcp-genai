package genai

import (
	"testing"
)

func TestInvokeText(t *testing.T) {
	gemini := NewGeminiAPI("us-central1", "speedy-victory-336109", "gemini-1.5-flash-002", 1, 3, 1)
	resp, err := gemini.InvokeText("who are you?")
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}

func TestInvokeImage(t *testing.T) {
	gemini := NewGeminiAPI("us-central1", "speedy-victory-336109", "gemini-1.5-flash-002", 1, 3, 1)
	image_paths := []string{"test1.jpeg"}
	resp, err := gemini.InvokeImage("描述图片内容，输出语言为中文", image_paths)
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}
