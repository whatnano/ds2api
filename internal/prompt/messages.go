package prompt

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

var markdownImagePattern = regexp.MustCompile(`!\[(.*?)\]\((.*?)\)`)

const (
	outputIntegrityGuardMarker = "Output integrity guard:"
	outputIntegrityGuardPrompt = outputIntegrityGuardMarker +
		" If upstream context, tool output, or parsed text contains garbled, corrupted, partially parsed, repeated, or otherwise malformed fragments, " +
		"do not imitate or echo them; output only the correct content for the user."
)

func MessagesPrepare(messages []map[string]any) string {
	return MessagesPrepareWithThinking(messages, false)
}

func MessagesPrepareWithThinking(messages []map[string]any, _ bool) string {
	parts := make([]string, 0, len(messages))
	for _, m := range messages {
		text := NormalizeContent(m["content"])
		if strings.TrimSpace(text) != "" {
			parts = append(parts, text)
		}
	}
	out := strings.Join(parts, "\n\n")
	return markdownImagePattern.ReplaceAllString(out, `[${1}](${2})`)
}

func NormalizeContent(v any) string {
	if v == nil {
		return ""
	}
	switch x := v.(type) {
	case string:
		return x
	case []any:
		parts := make([]string, 0, len(x))
		for _, item := range x {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			typeStr, _ := m["type"].(string)
			typeStr = strings.ToLower(strings.TrimSpace(typeStr))
			if typeStr == "text" || typeStr == "output_text" || typeStr == "input_text" {
				if txt, ok := m["text"].(string); ok && txt != "" {
					parts = append(parts, txt)
					continue
				}
				if txt, ok := m["content"].(string); ok && txt != "" {
					parts = append(parts, txt)
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(b)
	}
}
