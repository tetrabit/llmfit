# llmfit-desktop KNOWLEDGE BASE

## OVERVIEW

Small Tauri desktop app that wraps `llmfit-core` with a static HTML/CSS/JS frontend and a handful of Tauri commands.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Tauri commands | `src/main.rs` | `get_system_specs`, `get_model_fits`, pull lifecycle |
| Frontend UI | `ui/app.js`, `ui/index.html`, `ui/styles.css` | No React/Vite here; plain static assets |
| App config | `tauri.conf.json` | `frontendDist` points at `./ui`, CSP is restrictive |
| Permissions/capabilities | `capabilities/default.json` | Tauri capability config |

## CONVENTIONS

- Keep the frontend dependency-free unless there is a strong reason; current UI is plain JS.
- Exposed Tauri commands mirror read-heavy `llmfit-core` operations plus Ollama pull actions.
- Respect the static CSP in `tauri.conf.json` when changing frontend behavior.

## ANTI-PATTERNS

- Assuming the desktop app shares the React dashboard codepath.
- Adding browser-side dependencies that require a bundler without updating Tauri build config.
- Expanding the command surface without matching Rust-side state management.

## PACKAGE GOTCHAS

- This package is not a default workspace member; many root `cargo` commands skip it unless explicitly targeted.
- UI assets ship from `./ui`, not from `llmfit-web/dist`.
- `src/main.rs` manually maps `llmfit-core` enums like `RunMode` and `InferenceRuntime`; core enum additions can break desktop until those match arms are updated.

## COMMANDS

```bash
cargo run -p llmfit-desktop
cargo check -p llmfit-desktop
```
