package genai

import (
	"testing"
	"time"
)

func TestGeminiInvokeText(t *testing.T) {
	gemini := NewGeminiAPI("us-central1", "speedy-victory-336109", "gemini-1.5-flash-002", 1, 3, 1*time.Second)

	resp, err := gemini.Invoke(
		TextInput{Text: "who are you?"},
	)
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)

	// 测试多文本输入
	resp2, err := gemini.Invoke(
		TextInput{Text: "你好"},
		TextInput{Text: "请回答今天天气如何？"},
	)
	if err != nil {
		t.Error(err)
	}
	t.Log(resp2)
}

func TestGeminiInvokeImage(t *testing.T) {
	gemini := NewGeminiAPI("us-central1", "speedy-victory-336109", "gemini-1.5-flash-002", 1, 3, 1*time.Second)

	resp, err := gemini.Invoke(
		TextInput{Text: "一共有几张图片?"},
		TextInput{Text: "描述这个图片内容"},
		TextInput{Text: "输出语言为中文"},
		BlobInput{Path: "testdata/test1.jpeg"},
		BlobInput{Path: "testdata/test2.png"},
		BlobInput{Path: "testdata/test3.webp"},
	)
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}

func TestGeminiInvokeVideoGCS(t *testing.T) {
	gemini := NewGeminiAPI("us-central1", "speedy-victory-336109", "gemini-1.5-flash-002", 1, 3, 1*time.Second)

	resp, err := gemini.Invoke(
		TextInput{Text: "描述视频内容"},
		TextInput{Text: "用中文输出"},
		BlobInput{Path: "gs://pwm-lowa/videos/f4f4781e-6cd7-11ee-aae4-eedee28ea4dd.mp4"},
	)
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}

func TestGeminiInvokeVideoPublic(t *testing.T) {
	gemini := NewGeminiAPI("us-central1", "speedy-victory-336109", "gemini-1.5-flash-002", 1, 3, 1*time.Second)

	resp, err := gemini.Invoke(
		TextInput{Text: "描述视频内容"},
		TextInput{Text: "用中文输出"},
		BlobInput{Path: "https://storage.googleapis.com/pwm-lowa/videos/f4f4781e-6cd7-11ee-aae4-eedee28ea4dd.mp4"},
	)
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}

func TestGeminiInvokeAudio(t *testing.T) {
	gemini := NewGeminiAPI("us-central1", "speedy-victory-336109", "gemini-1.5-flash-002", 1, 3, 1*time.Second)

	resp, err := gemini.Invoke(
		TextInput{Text: "提取音频脚本"},
		TextInput{Text: "用中文输出"},
		BlobInput{Path: "testdata/test1.mp3"},
	)
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}

func TestGeminiIvokeVideo(t *testing.T) {
	gemini := NewGeminiAPI("us-central1", "speedy-victory-336109", "gemini-1.5-flash-002", 1, 3, 1*time.Second)
	resp, err := gemini.Invoke(
		TextInput{Text: "分别提取下列视频的脚本"},
		TextInput{Text: "用中文输出"},
		BlobInput{Path: "testdata/test1.mp4"},
		BlobInput{Path: "testdata/test2.webm"},
	)
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
	image_paths := []string{
		"testdata/test1.jpeg",
		"testdata/test2.png",
		"testdata/test3.webp",
	}
	text_prompts := []string{
		"一共有几张图片?",
		"描述图片内容",
		"输出语言为中文",
	}
	resp, err := claude.Invoke(text_prompts, image_paths)
	if err != nil {
		t.Error(err)
	}
	t.Log(resp)
}
