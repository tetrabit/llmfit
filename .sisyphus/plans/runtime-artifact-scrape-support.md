# Runtime Artifact Scrape Support Plan

## Goal

Extend the scraped/generated model data so base Hugging Face models can register runtime-compatible downloadable artifacts (primarily GGUF and MLX) discovered from quantized descendants, allowing at least 10 sampled models to become downloadable and runnable by an installed local runtime.

## Observed Seams

- `scripts/scrape_hf_models.py` generates JSON consumed by `llmfit-core/src/models.rs`.
- `gguf_sources` already exists in the embedded JSON schema and is used by `llmfit-tui/src/tui_app.rs` to surface llama.cpp download capability immediately.
- Provider downloadability otherwise depends on hardcoded mapping tables in `llmfit-core/src/providers.rs`.
- `update.rs` can merge cache models/overlays, but scraped artifact registration belongs earlier in the generated base data.

## Plan

1. Choose 10 sampled embedded models that currently lack usable downloadable runtime artifacts.
2. For each sampled model, discover runtime-compatible descendants from Hugging Face quantization/model-tree pages.
3. Prefer a durable scrape-time data path over hardcoded per-model provider mappings.
4. Extend `scripts/scrape_hf_models.py` to fetch and attach discovered `gguf_sources` for sampled models using authoritative external evidence.
5. Regenerate `llmfit-core/data/hf_models.json` and mirror output as required.
6. Verify that download capability now appears through existing llama.cpp logic without UI-specific hacks.
7. Add regression tests for artifact registration and representative sampled models.
8. Run validation and live-check the sampled models.

## Risks

- Hugging Face quantization pages are HTML-oriented and may be brittle to scrape.
- Descendant repo naming is inconsistent; selection heuristics must avoid low-quality or irrelevant forks where possible.
- Canonical base model identity must remain stable; artifact registration should enrich, not replace, base entries.
- Sampled models should bias toward text-generation models compatible with installed runtimes.
