package convert

import "strings"

// Protocol represents a well-known LLM API protocol.
type Protocol int

const (
	ProtocolUnknown        Protocol = iota
	ProtocolAnthropic               // Anthropic Messages API
	ProtocolOpenAIChat              // OpenAI Chat Completions API
	ProtocolOpenAIResponses         // OpenAI Responses API
)

func (p Protocol) String() string {
	switch p {
	case ProtocolAnthropic:
		return "anthropic"
	case ProtocolOpenAIChat:
		return "openai"
	case ProtocolOpenAIResponses:
		return "responses"
	default:
		return "unknown"
	}
}

// Direction is request or response.
type Direction int

const (
	DirectionRequest  Direction = iota
	DirectionResponse
)

// uriTable maps URI suffix patterns to (Protocol when request, Protocol when response).
// Uses strings.Contains for matching to handle reverse proxy prefixes (/api/anthropic/v1/messages)
// and query strings (/v1/messages?stream=true).
var uriTable = map[string]struct{ Req, Resp Protocol }{
	"/v1/messages":         {ProtocolAnthropic, ProtocolAnthropic},
	"/v1/chat/completions": {ProtocolOpenAIChat, ProtocolOpenAIChat},
	"/v1/responses":        {ProtocolOpenAIResponses, ProtocolOpenAIResponses},
}

// detectByURI resolves the protocol from the GOST URI metadata.
// Used as an optional fallback when detectSource returns Unknown.
func detectByURI(uri string, dir Direction) Protocol {
	for pattern, entry := range uriTable {
		if !strings.Contains(uri, pattern) {
			continue
		}
		if dir == DirectionRequest {
			return entry.Req
		}
		return entry.Resp
	}
	return ProtocolUnknown
}

// resolveModelTarget returns the target model from the model map, or inputModel if no match.
// Used by conversion functions that need model rewriting without protocol resolution.
func resolveModelTarget(inputModel string, mm ModelMap) string {
	if inputModel == "" || mm == nil {
		return inputModel
	}
	bare := StripProviderPrefix(inputModel)
	if target, _, ok := mm.Apply(bare); ok {
		return target
	}
	if target, _, ok := mm.Apply(inputModel); ok {
		return target
	}
	return inputModel
}

// parseProtocol converts a protocol string to a Protocol value.
func parseProtocol(s string) Protocol {
	switch strings.ToLower(s) {
	case "anthropic":
		return ProtocolAnthropic
	case "openai":
		return ProtocolOpenAIChat
	case "responses":
		return ProtocolOpenAIResponses
	default:
		return ProtocolUnknown
	}
}

// oppositeProtocol returns the "other" protocol — used as the default when
// no model map is configured (always convert between Anthropic and OpenAI Chat).
func oppositeProtocol(p Protocol) Protocol {
	switch p {
	case ProtocolAnthropic:
		return ProtocolOpenAIChat
	case ProtocolOpenAIChat:
		return ProtocolAnthropic
	default:
		return ProtocolUnknown
	}
}

// resolveModel determines the target model and downstream protocol.
func resolveModel(inputModel string, inputProtocol Protocol, mm ModelMap) (targetModel string, downstreamProtocol Protocol) {
	if inputModel != "" {
		bare := StripProviderPrefix(inputModel)

		// The model matches a downstream target (e.g. response body has
		// the rewritten model name). Return the declared protocol from the
		// model-map entry.
		if lp := mm.lookupTargetProtocol(bare); lp != "" {
			return inputModel, parseProtocol(lp)
		}
		if lp := mm.lookupTargetProtocol(inputModel); lp != "" {
			return inputModel, parseProtocol(lp)
		}

		if target, proto, ok := mm.Apply(bare); ok {
			targetModel = target
			if proto == "" {
				downstreamProtocol = inputProtocol
			} else {
				downstreamProtocol = parseProtocol(proto)
			}
			return
		}
		if target, proto, ok := mm.Apply(inputModel); ok {
			targetModel = target
			if proto == "" {
				downstreamProtocol = inputProtocol
			} else {
				downstreamProtocol = parseProtocol(proto)
			}
			return
		}

		// Model doesn't match any entry → convert to opposite protocol.
		return inputModel, oppositeProtocol(inputProtocol)
	}
	return inputModel, oppositeProtocol(inputProtocol)
}
