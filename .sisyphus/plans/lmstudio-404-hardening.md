# LM Studio 404 Hardening Plan

## Goal

Eliminate user-visible LM Studio download 404 failures by automatically trying robust ordered model identifier candidates before surfacing any error.

## Approach

1. Generate an ordered candidate list for LM Studio downloads instead of one tag.
2. Try candidates in this order:
   - explicit `lmstudio-community/*` catalog repo from `gguf_sources`
   - stripped repo name from that catalog repo
   - known LM Studio mapping from `lmstudio_pull_tag()`
   - stripped basename of mapped value if owner-prefixed
   - raw HF repo id
   - stripped basename of raw HF repo id
3. In `LmStudioProvider`, treat model-specific HTTP 404 / not-found initiation errors as retryable across candidates.
4. Only fall back to CLI after HTTP candidates are exhausted, and prefer the best native LM Studio key candidate for CLI.
5. Only surface an error after all HTTP candidates and viable CLI fallback paths fail.

## Tests

1. First candidate 404s, second succeeds over HTTP.
2. Owner-prefixed candidate 404s, stripped basename succeeds.
3. Mapping candidate succeeds before raw HF fallback is attempted.
4. All HTTP candidates fail, CLI candidate succeeds without surfacing 404 text.
5. Final error aggregates attempted candidates but does not emit misleading "API unavailable" wording for plain 404s.

## Acceptance Criteria

- Users no longer see the current `LM Studio API unavailable (http status: 404)` message for recoverable identifier mismatches.
- `start_lmstudio_download()` initiates ordered retries automatically.
- LM Studio tests prove retries happen before any error is surfaced.
- `cargo test -p llmfit lmstudio_tui_flow`, `cargo test -p llmfit-core lmstudio`, and `cargo check -p llmfit` all pass.
