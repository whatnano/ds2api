package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"ds2api/internal/auth"
	"ds2api/internal/promptcompat"
)

func historySplitTestMessages() []any {
	toolCalls := []any{
		map[string]any{
			"name":      "search",
			"arguments": map[string]any{"query": "docs"},
		},
	}
	return []any{
		map[string]any{"role": "system", "content": "system instructions"},
		map[string]any{"role": "user", "content": "first user turn"},
		map[string]any{
			"role":              "assistant",
			"content":           "",
			"reasoning_content": "hidden reasoning",
			"tool_calls":        toolCalls,
		},
		map[string]any{
			"role":         "tool",
			"name":         "search",
			"tool_call_id": "call-1",
			"content":      "tool result",
		},
		map[string]any{"role": "user", "content": "latest user turn"},
	}
}

func TestBuildOpenAICurrentInputContextTranscriptUsesNumberedHistorySections(t *testing.T) {
	transcript := buildOpenAICurrentInputContextTranscript(historySplitTestMessages())

	if strings.Contains(transcript, "[file content end]") || strings.Contains(transcript, "[file content begin]") || strings.Contains(transcript, "[file name]:") {
		t.Fatalf("expected transcript without file wrapper tags, got %q", transcript)
	}
	for _, want := range []string{
		"# DS2API_HISTORY.txt",
		"Prior conversation history and tool progress.",
		"=== 1. SYSTEM ===",
		"=== 2. USER ===",
		"=== 3. ASSISTANT ===",
		"=== 4. TOOL ===",
		"=== 5. USER ===",
		"first user turn",
		"tool result",
		"latest user turn",
		"[reasoning_content]",
		"hidden reasoning",
		"<|DSML|tool_calls>",
	} {
		if !strings.Contains(transcript, want) {
			t.Fatalf("expected transcript to contain %q, got %q", want, transcript)
		}
	}
}

func TestApplyCurrentInputFileLeavesPromptInline(t *testing.T) {
	ds := &inlineUploadDSStub{}
	h := &openAITestSurface{
		Store: mockOpenAIConfig{
			currentInputEnabled: true,
			currentInputMin:     0,
		},
		DS: ds,
	}
	req := map[string]any{
		"model":    "deepseek-v4-flash",
		"messages": historySplitTestMessages(),
	}
	stdReq, err := promptcompat.NormalizeOpenAIChatRequest(h.Store, req, "")
	if err != nil {
		t.Fatalf("normalize failed: %v", err)
	}

	out, err := h.applyCurrentInputFile(context.Background(), &auth.RequestAuth{DeepSeekToken: "token"}, stdReq)
	if err != nil {
		t.Fatalf("apply current input file failed: %v", err)
	}
	if len(ds.uploadCalls) != 0 {
		t.Fatalf("expected no automatic context uploads, got %d", len(ds.uploadCalls))
	}
	if out.CurrentInputFileApplied || out.HistoryText != "" || out.CurrentInputFileID != "" || out.CurrentToolsFileID != "" {
		t.Fatalf("expected current input file state to stay empty, got %#v", out)
	}
	if out.FinalPrompt != stdReq.FinalPrompt {
		t.Fatalf("expected prompt unchanged, got %q want %q", out.FinalPrompt, stdReq.FinalPrompt)
	}
	if !strings.Contains(out.FinalPrompt, "first user turn") || !strings.Contains(out.FinalPrompt, "latest user turn") {
		t.Fatalf("expected original prompt context to stay inline, got %q", out.FinalPrompt)
	}
}

func TestApplyThinkingInjectionLeavesLatestUserPromptUnchanged(t *testing.T) {
	enabled := true
	h := &openAITestSurface{
		Store: mockOpenAIConfig{
			thinkingInjection: &enabled,
			thinkingPrompt:    "custom thinking format",
		},
		DS: &inlineUploadDSStub{},
	}
	req := map[string]any{
		"model":    "deepseek-v4-flash",
		"messages": []any{map[string]any{"role": "user", "content": "hello"}},
	}
	stdReq, err := promptcompat.NormalizeOpenAIChatRequest(h.Store, req, "")
	if err != nil {
		t.Fatalf("normalize failed: %v", err)
	}

	out, err := h.applyCurrentInputFile(context.Background(), &auth.RequestAuth{DeepSeekToken: "token"}, stdReq)
	if err != nil {
		t.Fatalf("apply current input file failed: %v", err)
	}
	if out.FinalPrompt != stdReq.FinalPrompt {
		t.Fatalf("expected prompt unchanged, got %q want %q", out.FinalPrompt, stdReq.FinalPrompt)
	}
	if strings.Contains(out.FinalPrompt, promptcompat.ThinkingInjectionMarker) || strings.Contains(out.FinalPrompt, "custom thinking format") {
		t.Fatalf("expected no thinking injection in prompt, got %q", out.FinalPrompt)
	}
}

func TestChatCompletionsDoesNotUploadAutomaticContextFiles(t *testing.T) {
	ds := &inlineUploadDSStub{}
	h := &openAITestSurface{
		Store: mockOpenAIConfig{
			currentInputEnabled: true,
		},
		Auth: streamStatusAuthStub{},
		DS:   ds,
	}
	reqBody, _ := json.Marshal(map[string]any{
		"model":    "deepseek-v4-flash",
		"messages": historySplitTestMessages(),
		"stream":   false,
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(string(reqBody)))
	req.Header.Set("Authorization", "Bearer direct-token")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	h.ChatCompletions(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	if len(ds.uploadCalls) != 0 {
		t.Fatalf("expected no automatic context uploads, got %d", len(ds.uploadCalls))
	}
	if ds.completionReq == nil {
		t.Fatal("expected completion payload to be captured")
	}
	promptText, _ := ds.completionReq["prompt"].(string)
	if strings.Contains(promptText, "Continue from the latest state in the attached DS2API_HISTORY.txt context.") || strings.Contains(promptText, "DS2API_TOOLS.txt") {
		t.Fatalf("expected no generated file continuation prompt, got %q", promptText)
	}
	if !strings.Contains(promptText, "first user turn") || !strings.Contains(promptText, "latest user turn") {
		t.Fatalf("expected original context inline, got %q", promptText)
	}
	refIDs, _ := ds.completionReq["ref_file_ids"].([]any)
	if len(refIDs) != 0 {
		t.Fatalf("expected no generated ref_file_ids, got %#v", refIDs)
	}
}
