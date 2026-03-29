# llmfit-web KNOWLEDGE BASE

## OVERVIEW

React + Vite dashboard for the local `llmfit serve` API; built assets are embedded into `llmfit-tui` at compile time.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| App shell + filter UX | `src/App.jsx` | Loads system/models, owns theme + selection state |
| API client/query building | `src/api.js` | Shapes `/api/v1/system` and `/api/v1/models` calls |
| Tests | `src/App.test.jsx`, `src/api.test.js` | Vitest + Testing Library |
| Styling | `src/styles.css` | Single large stylesheet |
| Dev/build/test config | `package.json`, `vite.config.js` | Proxy to `127.0.0.1:8787`, jsdom tests |

## CONVENTIONS

- Keep the frontend API contract aligned with `llmfit-tui/src/serve_api.rs`.
- Vite dev server proxies `/api` and `/health` to the Rust server.
- `too_tight` filtering is partly client-side; preserve that behavior when changing queries.
- Theme state is persisted in localStorage under `llmfit-theme`.

## ANTI-PATTERNS

- Hard-coding API hosts in app code; local dev relies on the Vite proxy.
- Assuming the frontend is deployed standalone; production assets are embedded into Rust.
- Moving test setup outside `src/test-setup.js` without updating Vite/Vitest config.

## PACKAGE GOTCHAS

- `npm run build` must succeed before `llmfit-tui/build.rs` can embed real assets.
- UI state is intentionally lightweight and local; there is no shared store.

## COMMANDS

```bash
npm ci
npm run dev
npm run build
npm test
```
