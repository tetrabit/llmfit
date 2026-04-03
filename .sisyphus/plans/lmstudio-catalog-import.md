# LM Studio Catalog Import Plan

## Goal

Extend `llmfit-core/src/update.rs` so `update_model_cache(...)` can discover models from `https://lmstudio.ai/models`, fetch concrete LM Studio detail pages, derive `LlmModel` entries from those pages, and merge them into the shared cache so they appear in the app.

## Constraints

- Reuse the existing shared cache path rather than inventing a separate model list pipeline.
- Preserve existing HF refresh/trending/download flows.
- Avoid replacing a stronger existing HF-backed entry with a weaker LM Studio-derived entry unless the record is genuinely new.
- Keep LM Studio overlays working for existing HF-backed models.
- Prefer detail-page-derived records only when the catalog index lacks enough structured data.

## Planned Steps

1. Add parsers/helpers in `llmfit-core/src/update.rs` for:
   - extracting LM Studio catalog detail links from the `/models` index
   - parsing LM Studio detail-page `model.yaml` content into a richer intermediate record
   - converting that record into `LlmModel` values

2. Extend `update_model_cache(...)` to:
   - fetch the LM Studio catalog index
   - crawl candidate detail pages
   - upsert discovered `LlmModel`s into the shared cache
   - save any overlay metadata for matching existing HF-backed entries

3. Add merge guards so LM Studio catalog imports do not clobber better existing cached HF entries accidentally.

4. Add regression tests for:
   - catalog link extraction
   - detail-page parsing
   - LM Studio catalog record → `LlmModel` mapping
   - merge behavior for new catalog-only entries

5. Run:
   - `cargo check -p llmfit-core`
   - `cargo test -p llmfit-core`
   - `cargo check -p llmfit`

## Risks to Watch

- Catalog index may contain family pages instead of concrete model pages.
- Detail pages may only expose LM Studio artifact repos, not canonical upstream HF repos.
- Duplicate identity may differ between LM Studio slugs and HF repo IDs.
- Some catalog records may be incomplete and require conservative defaults.
