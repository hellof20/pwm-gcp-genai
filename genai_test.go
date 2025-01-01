package genai

import (
	"testing"
)

func TestGeminiInvokeText(t *testing.T) {
	gemini := NewGeminiAPI("us-central1", "speedy-victory-336109", "gemini-1.5-flash-002", 1, 3, 1)
	text_prompts := []string{"who are you?"}
	resp, err := gemini.Invoke(text_prompts, []string{})
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}

func TestGeminiInvokeImage(t *testing.T) {
	gemini := NewGeminiAPI("us-central1", "speedy-victory-336109", "gemini-1.5-flash-002", 1, 3, 1)
	image_paths := []string{"test1.jpeg", "test2.png", "test3.webp"}
	text_prompts := []string{"一共有几张图片?", "描述这个图片内容", "输出语言为中文"}
	resp, err := gemini.Invoke(text_prompts, image_paths)
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}

func TestClaudeInvokeText(t *testing.T) {
	claude := NewClaudeAPI("us-east5", "speedy-victory-336109", "claude-3-5-sonnet@20240620", 1, 3, 1)
	text_prompts := []string{"who are you?"}
	resp, err := claude.Invoke(text_prompts, []string{})
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}

func TestClaudeInvokeImage(t *testing.T) {
	claude := NewClaudeAPI("us-east5", "speedy-victory-336109", "claude-3-5-sonnet@20240620", 1, 3, 1)
	image_paths := []string{"test1.jpeg", "test2.png", "test3.webp"}
	text_prompts := []string{"一共有几张图片?", "描述图片内容", "输出语言为中文"}
	resp, err := claude.Invoke(text_prompts, image_paths)
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}
