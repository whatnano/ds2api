package client

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

var clientSigningKey string

func init() {
	clientSigningKey = strings.TrimSpace(os.Getenv("DS2API_CLIENT_SIGNING_KEY"))
}

func signingEnabled() bool {
	return clientSigningKey != ""
}

// generateClientNonce returns a random 16-byte hex nonce.
func generateClientNonce() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// buildSignaturePayload constructs the canonical payload to sign:
//
//	method + "\n" + path + "\n" + timestamp + "\n" + nonce
func buildSignaturePayload(method, path string, timestamp int64, nonce string) string {
	return strings.ToUpper(strings.TrimSpace(method)) + "\n" +
		path + "\n" +
		strconv.FormatInt(timestamp, 10) + "\n" +
		nonce
}

// computeClientSignature returns HMAC-SHA256(key, payload) as hex.
func computeClientSignature(key, payload string) string {
	mac := hmac.New(sha256.New, []byte(key))
	mac.Write([]byte(payload))
	return hex.EncodeToString(mac.Sum(nil))
}

// injectClientSignatureHeaders adds X-Client-Nonce, X-Client-Timestamp,
// and X-Client-Signature to the request headers when DS2API_CLIENT_SIGNING_KEY
// is configured. The signature covers method + path + timestamp + nonce.
func injectClientSignatureHeaders(req *http.Request) {
	if req == nil || !signingEnabled() {
		return
	}
	nonce := generateClientNonce()
	ts := time.Now().UnixMilli()
	payload := buildSignaturePayload(req.Method, req.URL.Path, ts, nonce)
	sig := computeClientSignature(clientSigningKey, payload)

	req.Header.Set("X-Client-Nonce", nonce)
	req.Header.Set("X-Client-Timestamp", strconv.FormatInt(ts, 10))
	req.Header.Set("X-Client-Signature", sig)
}
