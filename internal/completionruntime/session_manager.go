package completionruntime

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"ds2api/internal/auth"
	"ds2api/internal/promptcompat"
)

const minPromptDedupePrefixRunes = 64

var apiSessions = newAPISessionManager()

type apiSessionManager struct {
	mu       sync.Mutex
	sessions map[string]*apiSessionEntry
}

type apiSessionEntry struct {
	mu                    sync.Mutex
	sessionID             string
	firstPrompt           string
	lastResponseMessageID int
	accountKey            string
	lastUsed              time.Time
}

type APISessionLease struct {
	entry       *apiSessionEntry
	locked      bool
	sessionID   string
	firstTurn   bool
	accountKey  string
	requestFull string
}

func newAPISessionManager() *apiSessionManager {
	return &apiSessionManager{sessions: map[string]*apiSessionEntry{}}
}

func acquireAPISessionLease(ctx context.Context, ds DeepSeekCaller, a *auth.RequestAuth, stdReq promptcompat.StandardRequest, maxAttempts int) (*APISessionLease, string, any, string, error) {
	key := sessionKey(a)
	if key == "" {
		sessionID, err := ds.CreateSession(ctx, a, maxAttempts)
		return nil, sessionID, nil, stdReq.FinalPrompt, err
	}
	accountKey := sessionAccountKey(a)
	entry := apiSessions.entry(key)
	entry.mu.Lock()
	lease := &APISessionLease{
		entry:       entry,
		locked:      true,
		accountKey:  accountKey,
		requestFull: stdReq.FinalPrompt,
	}
	if entry.accountKey != "" && entry.accountKey != accountKey {
		entry.clear()
	}
	entry.accountKey = accountKey
	if entry.sessionID == "" || entry.lastResponseMessageID <= 0 {
		sessionID, err := ds.CreateSession(ctx, a, maxAttempts)
		if err != nil {
			lease.release()
			return nil, "", nil, "", err
		}
		entry.sessionID = sessionID
		entry.firstPrompt = stdReq.FinalPrompt
		entry.lastResponseMessageID = 0
		entry.lastUsed = time.Now()
		lease.sessionID = sessionID
		lease.firstTurn = true
		return lease, sessionID, nil, stdReq.FinalPrompt, nil
	}
	lease.sessionID = entry.sessionID
	prompt := dedupePromptPrefix(stdReq.FinalPrompt, entry.firstPrompt)
	entry.lastUsed = time.Now()
	return lease, entry.sessionID, entry.lastResponseMessageID, prompt, nil
}

func (m *apiSessionManager) entry(key string) *apiSessionEntry {
	m.mu.Lock()
	defer m.mu.Unlock()
	entry := m.sessions[key]
	if entry == nil {
		entry = &apiSessionEntry{}
		m.sessions[key] = entry
	}
	return entry
}

func (l *APISessionLease) complete(responseMessageID int) {
	if l == nil || !l.locked {
		return
	}
	if responseMessageID > 0 && l.entry != nil {
		l.entry.sessionID = l.sessionID
		l.entry.accountKey = l.accountKey
		if l.firstTurn || strings.TrimSpace(l.entry.firstPrompt) == "" {
			l.entry.firstPrompt = l.requestFull
		}
		l.entry.lastResponseMessageID = responseMessageID
		l.entry.lastUsed = time.Now()
	} else if l.entry != nil {
		l.entry.clear()
	}
	l.release()
}

func (l *APISessionLease) invalidateAndRelease() {
	if l == nil {
		return
	}
	if l.entry != nil {
		l.entry.clear()
	}
	l.release()
}

func (l *APISessionLease) release() {
	if l == nil || !l.locked {
		return
	}
	l.locked = false
	l.entry.mu.Unlock()
}

func ReleaseSessionLease(l *APISessionLease) {
	if l != nil {
		l.release()
	}
}

func (e *apiSessionEntry) clear() {
	e.sessionID = ""
	e.firstPrompt = ""
	e.lastResponseMessageID = 0
	e.accountKey = ""
	e.lastUsed = time.Time{}
}

func sessionKey(a *auth.RequestAuth) string {
	if a == nil {
		return ""
	}
	return strings.TrimSpace(a.CallerID)
}

func sessionAccountKey(a *auth.RequestAuth) string {
	if a == nil {
		return ""
	}
	if strings.TrimSpace(a.AccountID) != "" {
		return "account:" + strings.TrimSpace(a.AccountID)
	}
	if strings.TrimSpace(a.DeepSeekToken) != "" {
		sum := sha256.Sum256([]byte(strings.TrimSpace(a.DeepSeekToken)))
		return "token:" + hex.EncodeToString(sum[:8])
	}
	return "direct"
}

func dedupePromptPrefix(current, first string) string {
	if current == "" || first == "" {
		return current
	}
	commonBytes := longestCommonPrefixBytes(current, first)
	if commonBytes == 0 {
		return current
	}
	if utf8.RuneCountInString(current[:commonBytes]) < minPromptDedupePrefixRunes {
		return current
	}
	rest := strings.TrimLeft(current[commonBytes:], " \t\r\n")
	if rest == "" {
		return current
	}
	return rest
}

func longestCommonPrefixBytes(a, b string) int {
	i := 0
	for i < len(a) && i < len(b) {
		ra, sizeA := utf8.DecodeRuneInString(a[i:])
		rb, sizeB := utf8.DecodeRuneInString(b[i:])
		if ra != rb || sizeA != sizeB {
			break
		}
		i += sizeA
	}
	return i
}

func isContextFullError(status int, message string) bool {
	if status == http.StatusOK {
		return false
	}
	lower := strings.ToLower(strings.TrimSpace(message))
	if lower == "" {
		return false
	}
	if strings.Contains(lower, "\u4e0a\u4e0b\u6587") || strings.Contains(lower, "context") {
		return strings.Contains(lower, "full") ||
			strings.Contains(lower, "length") ||
			strings.Contains(lower, "window") ||
			strings.Contains(lower, "maximum") ||
			strings.Contains(lower, "max") ||
			strings.Contains(lower, "limit") ||
			strings.Contains(lower, "too long") ||
			strings.Contains(lower, "exceed") ||
			strings.Contains(lower, "\u6ee1")
	}
	return strings.Contains(lower, "token limit") ||
		strings.Contains(lower, "too many tokens") ||
		strings.Contains(lower, "prompt is too long")
}
