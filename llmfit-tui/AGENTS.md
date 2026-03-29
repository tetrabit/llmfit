# llmfit-tui KNOWLEDGE BASE

## OVERVIEW

Main product crate: clap CLI, terminal UI, embedded web dashboard server, and glue to `llmfit-core`.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| CLI flags / subcommands | `src/main.rs` | Large entrypoint; also owns env loading + dashboard auto-start |
| TUI state + filters | `src/tui_app.rs` | All app mutation lives here |
| Key handling | `src/tui_events.rs` | Mode-specific input handling |
| Rendering | `src/tui_ui.rs` | Stateless draw functions only |
| CLI tables / JSON helpers | `src/display.rs` | Classic non-TUI output path |
| REST API + SPA serving | `src/serve_api.rs` | Axum routes for `/health` and `/api/v1/*` |
| Web embed step | `build.rs` | Embeds `../llmfit-web/dist`; fallback page if missing |

## CONVENTIONS

- Keep CLI-mode code independent from TUI state.
- `tui_ui.rs` may take `&mut App` for widget state, not business-state mutation.
- Add filters in four places: state, filter application, key handling, UI, then help text.
- `serve_api.rs` serves both JSON endpoints and static frontend assets; preserve both when changing routes.

## ANTI-PATTERNS

- Mutating `App` during rendering.
- Adding a subcommand without wiring clap docs and dispatch in `main.rs`.
- Forgetting that web assets are generated at build time, not loaded from disk at runtime.
- Breaking `--json` behavior for commands intended for tool/agent use.

## PACKAGE GOTCHAS

- `src/main.rs` and `src/serve_api.rs` both contain tests near the end of the file.
- CI builds `llmfit-web` first so `llmfit-tui/build.rs` sees a real `dist`; local Rust builds may hit the fallback asset path instead.
- The default binary/package name is `llmfit`, not `llmfit-tui`.

## COMMANDS

```bash
cargo run -p llmfit -- --cli
cargo run -p llmfit -- serve --host 127.0.0.1 --port 8787
cargo test -p llmfit
cargo check -p llmfit
```
