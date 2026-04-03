# LM Studio Download Test Plan

## Goal

Add robust automated coverage for LM Studio downloads triggered through the TUI app flow, with special focus on HTTP 404 handling and CLI fallback behavior.

## Test seam

- Trigger download through `App::start_lmstudio_download(...)`
- Observe progress/completion through `App::tick_pull()` and `pull_status`
- Simulate LM Studio API with a tiny local test HTTP server
- Simulate `lms` CLI with a temporary executable injected via `PATH`

## Required code changes

1. Factor LM Studio provider download logic so API and CLI fallback paths are testable.
2. Fall back to CLI not only when API is unreachable, but also when the API responds with model-specific failures such as 404.
3. Add TUI tests that exercise the real TUI-triggered path programmatically.

## Minimum scenarios

1. HTTP success with progress polling to completion.
2. HTTP `already_downloaded` immediate success.
3. HTTP 404 for model tag with CLI fallback success.
4. HTTP 404 with CLI fallback failure surfaced to TUI status.
5. Direct CLI path when API is unavailable and CLI is allowed.

## Acceptance criteria

- Five automated tests pass.
- Tests assert final TUI-visible outcomes (`pull_status`, `pull_active`, selected tag behavior).
- No real LM Studio installation or interactive TUI required.
- `cargo test -p llmfit` and `cargo check -p llmfit` pass.
