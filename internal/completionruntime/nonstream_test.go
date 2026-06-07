package completionruntime

import (
	"context"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"ds2api/internal/account"
	"ds2api/internal/assistantturn"
	"ds2api/internal/auth"
	"ds2api/internal/config"
	dsclient "ds2api/internal/deepseek/client"
	"ds2api/internal/promptcompat"
)

type fakeDeepSeekCaller struct {
	responses          []*http.Response
	payloads           []map[string]any
	uploads            []dsclient.UploadFileRequest
	completionAccounts []string
	createdSessions    []string
	sessionByAccount   bool
}

type currentInputRuntimeConfig struct{}

func (currentInputRuntimeConfig) CurrentInputFileEnabled() bool { return true }
func (currentInputRuntimeConfig) CurrentInputFileMinChars() int { return 0 }

func (f *fakeDeepSeekCaller) CreateSession(_ context.Context, a *auth.RequestAuth, _ int) (string, error) {
	if f.sessionByAccount && a != nil && a.AccountID != "" {
		sessionID := "session-" + a.AccountID
		f.createdSessions = append(f.createdSessions, sessionID)
		return sessionID, nil
	}
	sessionID := "session-1"
	if len(f.createdSessions) > 0 {
		sessionID = "session-" + string(rune('1'+len(f.createdSessions)))
	}
	f.createdSessions = append(f.createdSessions, sessionID)
	return sessionID, nil
}

func (f *fakeDeepSeekCaller) GetPow(context.Context, *auth.RequestAuth, int) (string, error) {
	return "pow", nil
}

func (f *fakeDeepSeekCaller) UploadFile(_ context.Context, a *auth.RequestAuth, req dsclient.UploadFileRequest, _ int) (*dsclient.UploadFileResult, error) {
	f.uploads = append(f.uploads, req)
	if a != nil && a.AccountID != "" {
		return &dsclient.UploadFileResult{ID: "file-runtime-" + a.AccountID}, nil
	}
	return &dsclient.UploadFileResult{ID: "file-runtime-1"}, nil
}

func (f *fakeDeepSeekCaller) CallCompletion(_ context.Context, a *auth.RequestAuth, payload map[string]any, _ string, _ int) (*http.Response, error) {
	f.payloads = append(f.payloads, payload)
	if a != nil {
		f.completionAccounts = append(f.completionAccounts, a.AccountID)
	}
	if len(f.responses) == 0 {
		return sseHTTPResponse(http.StatusOK, `data: {"p":"response/content","v":"fallback"}`), nil
	}
	resp := f.responses[0]
	f.responses = f.responses[1:]
	return resp, nil
}

type concurrentSessionCaller struct {
	mu              sync.Mutex
	payloads        []map[string]any
	createdSessions int
	firstStarted    chan struct{}
	secondStarted   chan struct{}
	releaseFirst    chan struct{}
}

func newConcurrentSessionCaller() *concurrentSessionCaller {
	return &concurrentSessionCaller{
		firstStarted:  make(chan struct{}),
		secondStarted: make(chan struct{}),
		releaseFirst:  make(chan struct{}),
	}
}

func (c *concurrentSessionCaller) CreateSession(context.Context, *auth.RequestAuth, int) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.createdSessions++
	return "session-1", nil
}

func (c *concurrentSessionCaller) GetPow(context.Context, *auth.RequestAuth, int) (string, error) {
	return "pow", nil
}

func (c *concurrentSessionCaller) UploadFile(context.Context, *auth.RequestAuth, dsclient.UploadFileRequest, int) (*dsclient.UploadFileResult, error) {
	return &dsclient.UploadFileResult{ID: "file-id"}, nil
}

func (c *concurrentSessionCaller) CallCompletion(_ context.Context, _ *auth.RequestAuth, payload map[string]any, _ string, _ int) (*http.Response, error) {
	c.mu.Lock()
	c.payloads = append(c.payloads, payload)
	callIndex := len(c.payloads)
	c.mu.Unlock()
	if callIndex == 1 {
		close(c.firstStarted)
		<-c.releaseFirst
		return sseHTTPResponse(http.StatusOK, `data: {"response_message_id":701,"p":"response/content","v":"first"}`), nil
	}
	close(c.secondStarted)
	return sseHTTPResponse(http.StatusOK, `data: {"response_message_id":702,"p":"response/content","v":"second"}`), nil
}

func TestExecuteNonStreamWithRetryBuildsCanonicalTurn(t *testing.T) {
	ds := &fakeDeepSeekCaller{responses: []*http.Response{sseHTTPResponse(
		http.StatusOK,
		`data: {"response_message_id":42,"p":"response/content","v":"<tool_calls><invoke name=\"Write\"><parameter name=\"content\">{\"x\":1}</parameter></invoke></tool_calls>"}`,
	)}}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "prompt",
		FinalPrompt:     "final prompt",
		ToolNames:       []string{"Write"},
		ToolsRaw: []any{map[string]any{
			"name": "Write",
			"input_schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"content": map[string]any{"type": "string"},
				},
			},
		}},
	}

	result, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, &auth.RequestAuth{}, stdReq, Options{})
	if outErr != nil {
		t.Fatalf("unexpected output error: %#v", outErr)
	}
	if result.SessionID != "session-1" {
		t.Fatalf("session mismatch: %q", result.SessionID)
	}
	if got := result.Turn.ResponseMessageID; got != 42 {
		t.Fatalf("response message id mismatch: %d", got)
	}
	if len(result.Turn.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(result.Turn.ToolCalls))
	}
	if _, ok := result.Turn.ToolCalls[0].Input["content"].(string); !ok {
		t.Fatalf("expected schema-normalized string argument, got %#v", result.Turn.ToolCalls[0].Input["content"])
	}
	if result.Turn.Usage.InputTokens == 0 || result.Turn.Usage.TotalTokens == 0 {
		t.Fatalf("expected usage to be populated, got %#v", result.Turn.Usage)
	}
}

func TestExecuteNonStreamWithRetrySwitchesManagedAccountBeforeFinal429(t *testing.T) {
	t.Setenv("DS2API_CONFIG_JSON", `{
		"keys":["managed-key"],
		"accounts":[
			{"email":"acc1@test.com","password":"pwd"},
			{"email":"acc2@test.com","password":"pwd"}
		]
	}`)
	store := config.LoadStore()
	resolver := auth.NewResolver(store, account.NewPool(store), func(_ context.Context, acc config.Account) (string, error) {
		return "token-" + acc.Identifier(), nil
	})
	req, _ := http.NewRequest(http.MethodPost, "/", nil)
	req.Header.Set("Authorization", "Bearer managed-key")
	a, err := resolver.Determine(req)
	if err != nil {
		t.Fatalf("determine failed: %v", err)
	}
	defer resolver.Release(a)

	ds := &fakeDeepSeekCaller{
		sessionByAccount: true,
		responses: []*http.Response{
			sseHTTPResponse(http.StatusOK, `data: {"response_message_id":11,"p":"response/thinking_content","v":"first empty"}`),
			sseHTTPResponse(http.StatusOK, `data: {"response_message_id":12,"p":"response/thinking_content","v":"retry empty"}`),
			sseHTTPResponse(http.StatusOK, `data: {"response_message_id":21,"p":"response/content","v":"ok from second account"}`),
		},
	}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "prompt",
		FinalPrompt:     "final prompt",
		Thinking:        true,
	}

	result, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, a, stdReq, Options{RetryEnabled: true})
	if outErr != nil {
		t.Fatalf("unexpected output error after account switch retry: %#v", outErr)
	}
	if result.Turn.Text != "ok from second account" {
		t.Fatalf("text mismatch after switch retry: %q", result.Turn.Text)
	}
	if result.SessionID != "session-acc2@test.com" {
		t.Fatalf("expected switched account session, got %q", result.SessionID)
	}
	wantAccounts := []string{"acc1@test.com", "acc1@test.com", "acc2@test.com"}
	if len(ds.completionAccounts) != len(wantAccounts) {
		t.Fatalf("completion account count mismatch: got %v want %v", ds.completionAccounts, wantAccounts)
	}
	for i, want := range wantAccounts {
		if ds.completionAccounts[i] != want {
			t.Fatalf("completion account %d = %q want %q (all=%v)", i, ds.completionAccounts[i], want, ds.completionAccounts)
		}
	}
	if got := ds.payloads[2]["chat_session_id"]; got != "session-acc2@test.com" {
		t.Fatalf("switched payload session mismatch: %#v", got)
	}
	if prompt, _ := ds.payloads[2]["prompt"].(string); strings.Contains(prompt, "Previous reply had no visible output") {
		t.Fatalf("expected fresh switched-account prompt without empty-output suffix, got %q", prompt)
	}
}

func TestExecuteNonStreamWithRetryDoesNotReuploadGeneratedContextAfterAccountSwitch(t *testing.T) {
	t.Setenv("DS2API_CONFIG_JSON", `{
		"keys":["managed-key"],
		"accounts":[
			{"email":"acc1@test.com","password":"pwd"},
			{"email":"acc2@test.com","password":"pwd"}
		]
	}`)
	store := config.LoadStore()
	resolver := auth.NewResolver(store, account.NewPool(store), func(_ context.Context, acc config.Account) (string, error) {
		return "token-" + acc.Identifier(), nil
	})
	req, _ := http.NewRequest(http.MethodPost, "/", nil)
	req.Header.Set("Authorization", "Bearer managed-key")
	a, err := resolver.Determine(req)
	if err != nil {
		t.Fatalf("determine failed: %v", err)
	}
	defer resolver.Release(a)

	ds := &fakeDeepSeekCaller{
		sessionByAccount: true,
		responses: []*http.Response{
			sseHTTPResponse(http.StatusOK, `data: {"response_message_id":11,"p":"response/thinking_content","v":"first empty"}`),
			sseHTTPResponse(http.StatusOK, `data: {"response_message_id":12,"p":"response/thinking_content","v":"retry empty"}`),
			sseHTTPResponse(http.StatusOK, `data: {"response_message_id":21,"p":"response/content","v":"ok from second account"}`),
		},
	}
	stdReq := promptcompat.StandardRequest{
		Surface:        "test",
		RequestedModel: "deepseek-v4-flash",
		ResolvedModel:  "deepseek-v4-flash",
		ResponseModel:  "deepseek-v4-flash",
		Messages: []any{
			map[string]any{"role": "user", "content": "large current input"},
		},
		PromptTokenText: "large current input",
		FinalPrompt:     "large current input",
		Thinking:        true,
	}

	result, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, a, stdReq, Options{
		RetryEnabled:     true,
		CurrentInputFile: currentInputRuntimeConfig{},
	})
	if outErr != nil {
		t.Fatalf("unexpected output error after account switch retry: %#v", outErr)
	}
	if result.Turn.Text != "ok from second account" {
		t.Fatalf("text mismatch after switch retry: %q", result.Turn.Text)
	}
	if len(ds.uploads) != 0 {
		t.Fatalf("expected no generated context uploads, got %d", len(ds.uploads))
	}
	refIDs, _ := ds.payloads[2]["ref_file_ids"].([]any)
	if len(refIDs) != 0 {
		t.Fatalf("expected no generated ref_file_ids, got %#v", ds.payloads[2]["ref_file_ids"])
	}
}

func TestExecuteNonStreamWithRetryUsesParentMessageForEmptyRetry(t *testing.T) {
	ds := &fakeDeepSeekCaller{responses: []*http.Response{
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":77,"p":"response/thinking_content","v":"plan"}`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":78,"p":"response/content","v":"ok"}`),
	}}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "prompt",
		FinalPrompt:     "final prompt",
	}

	result, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, &auth.RequestAuth{}, stdReq, Options{RetryEnabled: true})
	if outErr != nil {
		t.Fatalf("unexpected output error: %#v", outErr)
	}
	if result.Attempts != 1 {
		t.Fatalf("expected one retry, got %d", result.Attempts)
	}
	if len(ds.payloads) != 2 {
		t.Fatalf("expected two completion calls, got %d", len(ds.payloads))
	}
	if got := ds.payloads[1]["parent_message_id"]; got != 77 {
		t.Fatalf("retry parent_message_id mismatch: %#v", got)
	}
	if result.Turn.Text != "ok" {
		t.Fatalf("retry text mismatch: %q", result.Turn.Text)
	}
}

func TestExecuteNonStreamReusesSessionByCallerID(t *testing.T) {
	apiSessions = newAPISessionManager()
	ds := &fakeDeepSeekCaller{responses: []*http.Response{
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":101,"p":"response/content","v":"first"}`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":102,"p":"response/content","v":"second"}`),
	}}
	a := &auth.RequestAuth{CallerID: "caller:a", DeepSeekToken: "token-a"}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "prompt",
		FinalPrompt:     strings.Repeat("system prelude ", 8) + "first question",
	}

	if _, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, a, stdReq, Options{}); outErr != nil {
		t.Fatalf("first request failed: %#v", outErr)
	}
	stdReq.FinalPrompt = strings.Repeat("system prelude ", 8) + "second question"
	if _, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, a, stdReq, Options{}); outErr != nil {
		t.Fatalf("second request failed: %#v", outErr)
	}
	if len(ds.createdSessions) != 1 {
		t.Fatalf("expected one session, got %v", ds.createdSessions)
	}
	if got := ds.payloads[1]["chat_session_id"]; got != ds.payloads[0]["chat_session_id"] {
		t.Fatalf("expected reused session, got first=%#v second=%#v", ds.payloads[0]["chat_session_id"], got)
	}
	if got := ds.payloads[1]["parent_message_id"]; got != 101 {
		t.Fatalf("expected parent id 101, got %#v", got)
	}
	if prompt, _ := ds.payloads[1]["prompt"].(string); strings.Contains(prompt, "system prelude") || prompt != "second question" {
		t.Fatalf("expected deduped prompt, got %q", prompt)
	}
}

func TestExecuteNonStreamSeparatesSessionsByCallerID(t *testing.T) {
	apiSessions = newAPISessionManager()
	ds := &fakeDeepSeekCaller{responses: []*http.Response{
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":201,"p":"response/content","v":"a"}`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":301,"p":"response/content","v":"b"}`),
	}}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "prompt",
		FinalPrompt:     "hello",
	}

	if _, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, &auth.RequestAuth{CallerID: "caller:a", DeepSeekToken: "token-a"}, stdReq, Options{}); outErr != nil {
		t.Fatalf("caller a failed: %#v", outErr)
	}
	if _, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, &auth.RequestAuth{CallerID: "caller:b", DeepSeekToken: "token-a"}, stdReq, Options{}); outErr != nil {
		t.Fatalf("caller b failed: %#v", outErr)
	}
	if len(ds.createdSessions) != 2 {
		t.Fatalf("expected two independent sessions, got %v", ds.createdSessions)
	}
	if ds.payloads[0]["chat_session_id"] == ds.payloads[1]["chat_session_id"] {
		t.Fatalf("expected distinct sessions, got %#v", ds.payloads)
	}
}

func TestExecuteNonStreamCreatesNewSessionAfterContextFull(t *testing.T) {
	apiSessions = newAPISessionManager()
	ds := &fakeDeepSeekCaller{responses: []*http.Response{
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":401,"p":"response/content","v":"first"}`),
		sseHTTPResponse(http.StatusBadRequest, `context length exceeded`),
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":501,"p":"response/content","v":"after reset"}`),
	}}
	a := &auth.RequestAuth{CallerID: "caller:a", DeepSeekToken: "token-a"}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "prompt",
		FinalPrompt:     strings.Repeat("system prelude ", 8) + "first question",
	}
	if _, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, a, stdReq, Options{}); outErr != nil {
		t.Fatalf("first request failed: %#v", outErr)
	}
	stdReq.FinalPrompt = strings.Repeat("system prelude ", 8) + "second question"
	result, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, a, stdReq, Options{})
	if outErr != nil {
		t.Fatalf("context-full retry failed: %#v", outErr)
	}
	if result.Turn.Text != "after reset" {
		t.Fatalf("unexpected retry text: %q", result.Turn.Text)
	}
	if len(ds.createdSessions) != 2 {
		t.Fatalf("expected new session after context full, got %v", ds.createdSessions)
	}
	if got := ds.payloads[2]["parent_message_id"]; got != nil {
		t.Fatalf("expected reset request without parent, got %#v", got)
	}
	if prompt, _ := ds.payloads[2]["prompt"].(string); prompt != stdReq.FinalPrompt {
		t.Fatalf("expected full prompt after reset, got %q", prompt)
	}
}

func TestExecuteNonStreamContextFullRetryOnlyOnce(t *testing.T) {
	apiSessions = newAPISessionManager()
	ds := &fakeDeepSeekCaller{responses: []*http.Response{
		sseHTTPResponse(http.StatusOK, `data: {"response_message_id":601,"p":"response/content","v":"first"}`),
		sseHTTPResponse(http.StatusBadRequest, `context window is full`),
		sseHTTPResponse(http.StatusBadRequest, `context window is full`),
	}}
	a := &auth.RequestAuth{CallerID: "caller:a", DeepSeekToken: "token-a"}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "prompt",
		FinalPrompt:     "hello",
	}
	if _, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, a, stdReq, Options{}); outErr != nil {
		t.Fatalf("first request failed: %#v", outErr)
	}
	_, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, a, stdReq, Options{})
	if outErr == nil {
		t.Fatal("expected context-full retry failure")
	}
	if len(ds.createdSessions) != 2 {
		t.Fatalf("expected exactly one recreated session, got %v", ds.createdSessions)
	}
	if len(ds.payloads) != 3 {
		t.Fatalf("expected three completion attempts, got %d", len(ds.payloads))
	}
}

func TestExecuteNonStreamSerializesConcurrentRequestsForCallerID(t *testing.T) {
	apiSessions = newAPISessionManager()
	ds := newConcurrentSessionCaller()
	a := &auth.RequestAuth{CallerID: "caller:a", DeepSeekToken: "token-a"}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "prompt",
		FinalPrompt:     strings.Repeat("system prelude ", 8) + "question",
	}

	errCh := make(chan *assistantturn.OutputError, 2)
	go func() {
		_, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, a, stdReq, Options{})
		errCh <- outErr
	}()
	<-ds.firstStarted

	go func() {
		nextReq := stdReq
		nextReq.FinalPrompt = strings.Repeat("system prelude ", 8) + "next question"
		_, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, a, nextReq, Options{})
		errCh <- outErr
	}()

	select {
	case <-ds.secondStarted:
		t.Fatal("second request reached DeepSeek before first response completed")
	case <-time.After(50 * time.Millisecond):
	}
	close(ds.releaseFirst)
	for i := 0; i < 2; i++ {
		if outErr := <-errCh; outErr != nil {
			t.Fatalf("request %d failed: %#v", i, outErr)
		}
	}
	ds.mu.Lock()
	defer ds.mu.Unlock()
	if ds.createdSessions != 1 {
		t.Fatalf("expected one shared session, got %d", ds.createdSessions)
	}
	if len(ds.payloads) != 2 {
		t.Fatalf("expected two payloads, got %d", len(ds.payloads))
	}
	if got := ds.payloads[1]["parent_message_id"]; got != 701 {
		t.Fatalf("expected serialized second parent id 701, got %#v", got)
	}
}

func TestExecuteNonStreamWithRetryConvertsReferenceMarkers(t *testing.T) {
	ds := &fakeDeepSeekCaller{responses: []*http.Response{sseHTTPResponse(
		http.StatusOK,
		`data: {"p":"response/content","v":"答案[reference:0]。","citation":{"cite_index":0,"url":"https://example.com/ref"}}`,
	)}}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test",
		ResponseModel:   "deepseek-v4-flash-search",
		PromptTokenText: "prompt",
		FinalPrompt:     "final prompt",
		Search:          true,
	}

	result, outErr := ExecuteNonStreamWithRetry(context.Background(), ds, &auth.RequestAuth{}, stdReq, Options{})
	if outErr != nil {
		t.Fatalf("unexpected output error: %#v", outErr)
	}
	want := "答案[0](https://example.com/ref)。"
	if result.Turn.Text != want {
		t.Fatalf("text mismatch: got %q want %q", result.Turn.Text, want)
	}
}

func TestStartCompletionLeavesCurrentInputInline(t *testing.T) {
	ds := &fakeDeepSeekCaller{responses: []*http.Response{sseHTTPResponse(http.StatusOK, `data: {"p":"response/content","v":"ok"}`)}}
	stdReq := promptcompat.StandardRequest{
		Surface:         "test_adapter",
		RequestedModel:  "deepseek-v4-flash",
		ResolvedModel:   "deepseek-v4-flash",
		ResponseModel:   "deepseek-v4-flash",
		PromptTokenText: "first user turn",
		FinalPrompt:     "first user turn",
		Messages: []any{
			map[string]any{"role": "user", "content": "first user turn"},
		},
	}

	start, outErr := StartCompletion(context.Background(), ds, &auth.RequestAuth{DeepSeekToken: "token"}, stdReq, Options{
		CurrentInputFile: currentInputRuntimeConfig{},
	})
	if outErr != nil {
		t.Fatalf("unexpected output error: %#v", outErr)
	}
	if len(ds.uploads) != 0 {
		t.Fatalf("expected no generated context uploads, got %d", len(ds.uploads))
	}
	if len(ds.payloads) != 1 {
		t.Fatalf("expected one completion payload, got %d", len(ds.payloads))
	}
	refIDs, _ := ds.payloads[0]["ref_file_ids"].([]any)
	if len(refIDs) != 0 {
		t.Fatalf("expected no generated ref_file_ids, got %#v", ds.payloads[0]["ref_file_ids"])
	}
	prompt, _ := ds.payloads[0]["prompt"].(string)
	if prompt != "first user turn" {
		t.Fatalf("expected original prompt, got %q", prompt)
	}
	if start.Request.CurrentInputFileApplied || start.Request.HistoryText != "" {
		t.Fatalf("expected no current input file state, got %#v", start.Request)
	}
}

func sseHTTPResponse(status int, lines ...string) *http.Response {
	body := strings.Join(lines, "\n")
	if !strings.HasSuffix(body, "\n") {
		body += "\n"
	}
	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}
