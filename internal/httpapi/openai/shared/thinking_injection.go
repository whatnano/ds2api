package shared

import "ds2api/internal/promptcompat"

func ApplyThinkingInjection(store ConfigReader, stdReq promptcompat.StandardRequest) promptcompat.StandardRequest {
	return stdReq
}
