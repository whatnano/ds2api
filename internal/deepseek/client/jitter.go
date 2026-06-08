package client

import (
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

var jitterMinMs, jitterMaxMs int

func init() {
	jitterMinMs, jitterMaxMs = parseJitterRange()
}

func parseJitterRange() (minMs, maxMs int) {
	raw := strings.TrimSpace(os.Getenv("DS2API_REQUEST_JITTER_MS"))
	if raw == "" {
		return 0, 0
	}
	parts := strings.SplitN(raw, "-", 2)
	if len(parts) == 1 {
		v, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil || v < 0 {
			return 0, 0
		}
		return 0, v
	}
	lo, err := strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil || lo < 0 {
		return 0, 0
	}
	hi, err := strconv.Atoi(strings.TrimSpace(parts[1]))
	if err != nil || hi < lo {
		return 0, 0
	}
	return lo, hi
}

func jitterEnabled() bool {
	return jitterMaxMs > 0
}

func applyRequestJitter() {
	if !jitterEnabled() {
		return
	}
	ms := jitterMinMs
	if jitterMaxMs > jitterMinMs {
		ms += rand.Intn(jitterMaxMs - jitterMinMs + 1)
	}
	if ms > 0 {
		time.Sleep(time.Duration(ms) * time.Millisecond)
	}
}
