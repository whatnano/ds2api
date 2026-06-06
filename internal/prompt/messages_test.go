package prompt

import (
	"strings"
	"testing"
)

func TestNormalizeContentNilReturnsEmpty(t *testing.T) {
	if got := NormalizeContent(nil); got != "" {
		t.Fatalf("expected empty string for nil content, got %q", got)
	}
}

func TestMessagesPrepareNilContentNoNullLiteral(t *testing.T) {
	messages := []map[string]any{
		{"role": "assistant", "content": nil},
		{"role": "user", "content": "ok"},
	}
	got := MessagesPrepare(messages)
	if got == "" {
		t.Fatalf("expected non-empty output")
	}
	if got == "null" {
		t.Fatalf("expected no null literal output, got %q", got)
	}
}

func TestMessagesPrepareUsesPlainTextOnly(t *testing.T) {
	messages := []map[string]any{
		{"role": "system", "content": "System rule"},
		{"role": "user", "content": "Question"},
		{"role": "assistant", "content": "Answer"},
	}
	got := MessagesPrepare(messages)
	if got != "System rule\n\nQuestion\n\nAnswer" {
		t.Fatalf("expected plain text transcript, got %q", got)
	}
	if containsChatTemplateMarker(got) {
		t.Fatalf("expected no chat-template markers, got %q", got)
	}
	if strings.Contains(got, "<think>") || strings.Contains(got, "</think>") {
		t.Fatalf("did not expect think tags in prompt, got %q", got)
	}
}

func TestMessagesPrepareSingleUserMessageIsExactText(t *testing.T) {
	const text = "你好，请用一句话介绍你自己。"
	got := MessagesPrepare([]map[string]any{{"role": "user", "content": text}})
	if got != text {
		t.Fatalf("expected exact user text, got %q", got)
	}
}

func TestMessagesPrepareDoesNotPrependOutputIntegrityGuard(t *testing.T) {
	messages := []map[string]any{
		{"role": "system", "content": "System rule"},
		{"role": "user", "content": "Question"},
	}
	got := MessagesPrepare(messages)
	if strings.Contains(got, outputIntegrityGuardPrompt) || strings.Contains(got, outputIntegrityGuardMarker) {
		t.Fatalf("expected no output integrity guard, got %q", got)
	}
	if !strings.HasPrefix(got, "System rule") {
		t.Fatalf("expected prompt to start with caller system prompt, got %q", got)
	}
	if !strings.Contains(got, "Question") {
		t.Fatalf("expected user question, got %q", got)
	}
	if containsChatTemplateMarker(got) {
		t.Fatalf("expected no chat-template markers, got %q", got)
	}
}

func TestNormalizeContentArrayFallsBackToContentWhenTextEmpty(t *testing.T) {
	got := NormalizeContent([]any{
		map[string]any{"type": "text", "text": "", "content": "from-content"},
	})
	if got != "from-content" {
		t.Fatalf("expected fallback to content when text is empty, got %q", got)
	}
}

func TestMessagesPrepareWithThinkingPreservesPromptShape(t *testing.T) {
	messages := []map[string]any{{"role": "user", "content": "Question"}}
	gotThinking := MessagesPrepareWithThinking(messages, true)
	gotPlain := MessagesPrepareWithThinking(messages, false)
	if gotThinking != gotPlain {
		t.Fatalf("expected thinking flag not to add extra continuity instructions, got thinking=%q plain=%q", gotThinking, gotPlain)
	}
	if containsChatTemplateMarker(gotThinking) {
		t.Fatalf("expected no chat-template markers, got %q", gotThinking)
	}
}

func containsChatTemplateMarker(s string) bool {
	for _, marker := range []string{
		"<|begin▁of▁sentence|>",
		"<|System|>",
		"<|User|>",
		"<|Assistant|>",
		"<|Tool|>",
		"<|end▁of▁sentence|>",
		"<|end▁of▁toolresults|>",
		"<|end▁of▁instructions|>",
	} {
		if strings.Contains(s, marker) {
			return true
		}
	}
	return false
}
