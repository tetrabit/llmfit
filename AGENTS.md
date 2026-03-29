# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-28 22:09 America/New_York
**Commit:** `7b40343`
**Branch:** `main`

## OVERVIEW

`llmfit` is a workspace, not a single crate. Rust packages live in `llmfit-core`, `llmfit-tui`, and `llmfit-desktop`; `llmfit-web` is a React/Vite dashboard embedded into the TUI server build.

## STRUCTURE

```text
llmfit/
├── llmfit-core/      # Shared fitting engine, hardware detection, providers, updates
├── llmfit-tui/       # CLI binary, TUI, REST API, web-asset embed build step
├── llmfit-web/       # React/Vite dashboard served by llmfit-tui
├── llmfit-desktop/   # Tauri desktop app using llmfit-core
├── scripts/          # Data regeneration + verification helpers
├── data/             # Mirrored/generated data for repo-level tooling/docs
└── .github/workflows/# CI builds web first, then Rust checks/tests
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Fit/scoring logic | `llmfit-core/src/fit.rs` | Runtime choice, fit levels, ranking, score components |
| Hardware detection | `llmfit-core/src/hardware.rs` | GPU backends, unified memory, cluster detection |
| Model DB + parsing | `llmfit-core/src/models.rs` | Embedded JSON comes from `llmfit-core/data/*.json` |
| Provider integration | `llmfit-core/src/providers.rs` | Ollama / llama.cpp / MLX / Docker MR / LM Studio |
| Online update/cache | `llmfit-core/src/update.rs` | Merges cache into embedded DB at startup |
| CLI subcommands | `llmfit-tui/src/main.rs` | Main binary, clap surface, TUI launch, dashboard auto-start |
| TUI state | `llmfit-tui/src/tui_app.rs` | Filters, selection, provider state, plan mode |
| TUI rendering | `llmfit-tui/src/tui_ui.rs` | Stateless drawing only |
| REST API + embedded web | `llmfit-tui/src/serve_api.rs`, `llmfit-tui/build.rs` | `llmfit-tui/build.rs` embeds `llmfit-web/dist` or a fallback page |
| Web dashboard | `llmfit-web/src/*` | React client for `/api/v1/*`; tests via Vitest |
| Desktop app | `llmfit-desktop/src/main.rs`, `llmfit-desktop/ui/*` | Tauri commands + static web UI |
| Data regeneration | `scripts/scrape_hf_models.py`, `scripts/scrape_docker_models.py` | Generates model JSON into package data |

## CONVENTIONS

- Workspace root `Cargo.toml` is only a workspace manifest; package-specific details live under each crate.
- Default workspace members are `llmfit-core` and `llmfit-tui`; desktop is separate work.
- No `unsafe` code.
- No `.unwrap()` on user-facing paths. `expect()` is acceptable only for internal invariants with a descriptive message.
- Fit remains VRAM-first; CPU/system RAM is fallback.
- Unified-memory systems skip split VRAM/RAM offload assumptions.
- TUI rendering must stay stateless; mutation belongs in `tui_app.rs` / `tui_events.rs`, not in drawing code.
- `display.rs` and TUI modules stay independent; CLI mode must work without TUI state.
- Generated model data is source-controlled. Regenerate via scripts; do not hand-edit JSON.

## ANTI-PATTERNS (THIS PROJECT)

- Re-introducing the old single-crate mental model (`src/*.rs` at repo root) — current code is package-scoped.
- Editing `llmfit-core/data/hf_models.json` or `llmfit-core/data/docker_models.json` by hand.
- Forgetting that `llmfit-tui/build.rs` depends on `llmfit-web/dist` and falls back when assets are missing.
- Mixing CLI rendering concerns into TUI modules, or mutating app state inside `tui_ui.rs`.
- Treating Apple unified memory like discrete GPU + RAM pools.

## UNIQUE STYLES

- Docs in `README.md` may lag behind workspace internals; verify current code paths before trusting old top-level descriptions.
- The HF scraper writes to both `data/hf_models.json` and `llmfit-core/data/hf_models.json`; runtime embedding uses the package-local file.
- CI always builds `llmfit-web` before Rust test/check/clippy jobs; only the `test` job runs `npm test`.

## COMMANDS

```bash
# Default-member Rust checks (root cargo commands skip llmfit-desktop)
cargo check --all-targets --all-features
cargo test --verbose
cargo fmt --all
cargo clippy --all-targets --all-features

# Full workspace / desktop-specific
cargo check --workspace --all-targets --all-features
cargo test --workspace --verbose
cargo check -p llmfit-desktop

# Main binary
cargo run -p llmfit -- --cli
cargo run -p llmfit -- serve --host 127.0.0.1 --port 8787

# Web dashboard
cd llmfit-web && npm ci && npm run build
cd llmfit-web && npm test

# Desktop app
cargo run -p llmfit-desktop

# Data refresh
python3 scripts/scrape_hf_models.py
python3 scripts/scrape_docker_models.py
python3 scripts/test_api.py --spawn
```

## NOTES

- LSP codemap was unavailable during generation (`rust-analyzer` missing), so file-level structure was derived from source/module reads.
- `scripts/update_models.sh` still reports root `data/hf_models.json`; runtime code embeds `llmfit-core/data/hf_models.json`, so keep both in sync.
- Child `AGENTS.md` files exist for package-local rules. Read the nearest one before editing.
