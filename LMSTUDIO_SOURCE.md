# LM Studio as a Model Metadata Source

## Goal

Add LM Studio as an additional source of model metadata so `llmfit` can:

- discover models that are prominent in the LM Studio catalog
- use LM Studio-specific metadata when it is more accurate for a runtime-ready artifact
- prefer local installed-artifact metadata over generic base-model metadata when appropriate
- reduce mismatches like:
  - Hugging Face model card says `262k`
  - LM Studio runtime/artifact reports `1,048,576`
  - `llmfit` currently only sees the Hugging Face side

This is not about replacing Hugging Face as the primary catalog. It is about adding a second source of truth for provider-specific artifact metadata.

## Current State

The codebase already has substantial LM Studio integration:

- local runtime detection and installed model listing in [providers.rs](/home/nullvoid/projects/llmfit/llmfit-core/src/providers.rs#L1296)
- LM Studio downloads from the TUI in [tui_app.rs](/home/nullvoid/projects/llmfit/llmfit-tui/src/tui_app.rs#L1963)
- name-matching helpers for installed LM Studio models in [providers.rs](/home/nullvoid/projects/llmfit/llmfit-core/src/providers.rs#L1596)

What is missing:

- LM Studio catalog ingestion
- LM Studio page or manifest metadata parsing
- precedence rules between:
  - embedded HF catalog
  - refreshed HF metadata
  - LM Studio catalog metadata
  - local installed artifact metadata

## Problem Statement

`llmfit` currently models a single `LlmModel` as if there were one canonical context length, capability set, and runtime profile per model.

That is too simple for LM Studio.

Examples of mismatch:

- base model metadata can differ from a runtime-specific GGUF artifact
- model-card prose can differ from runtime metadata or GGUF header metadata
- LM Studio may expose a context limit for a specific packaged artifact that differs from the base Hugging Face model card

The core issue is not scraping difficulty. The core issue is that we need to represent multiple metadata layers.

## Recommended Scope

Implement this in phases.

### Phase 1

Use LM Studio as a metadata overlay for installed LM Studio models only.

Outcome:

- if a model is installed in LM Studio and LM Studio exposes richer metadata, use that metadata in `llmfit`
- this solves the most user-visible mismatch first
- no full public LM Studio catalog scraping is required yet

### Phase 2

Use LM Studio public model pages as an optional online metadata source for known models.

Outcome:

- `llmfit --refresh-models` can augment HF metadata with LM Studio metadata
- this helps even when the model is not locally installed

### Phase 3

Add LM Studio catalog discovery as a first-class upstream source.

Outcome:

- `llmfit` can discover models from the LM Studio catalog, not only from Hugging Face
- catalog rankings and curated entries can influence discovery

## Data Model Changes

The current `LlmModel` struct is not enough for provider-specific metadata layering.

Recommended additions:

### 1. Introduce metadata provenance

Add a source/provenance enum, for example:

```rust
enum MetadataSource {
    EmbeddedHf,
    RefreshedHf,
    LmStudioCatalog,
    LocalLmStudioArtifact,
    LocalGgufArtifact,
}
```

### 2. Introduce overlay fields

Do not immediately replace the base model with LM Studio data. Keep base metadata plus optional overrides.

Possible shape:

```rust
struct ModelMetadataOverlay {
    source: MetadataSource,
    context_length: Option<u32>,
    capabilities: Option<Vec<Capability>>,
    use_case: Option<String>,
    format: Option<ModelFormat>,
    artifact_name: Option<String>,
    notes: Option<String>,
}
```

### 3. Add resolved/effective metadata helpers

Fit and UI logic should read a resolved view:

```rust
impl LlmModel {
    fn effective_context_length(&self) -> u32 { ... }
}
```

This avoids invasive changes across all TUI and scoring code.

## Source Precedence

This is the most important design rule.

Recommended precedence for fields like max context:

1. local installed artifact metadata
2. LM Studio catalog/artifact metadata
3. refreshed Hugging Face metadata
4. embedded catalog metadata
5. family heuristics

Why:

- local installed artifact is the closest thing to what the user will actually run
- LM Studio artifact metadata is often runtime-specific
- Hugging Face is still the best generic upstream source for base model identity
- family heuristics should remain a fallback, not a primary source

Important nuance:

- do not blindly overwrite everything from LM Studio
- prefer field-level precedence
- example:
  - LM Studio may be better for `context_length`
  - Hugging Face may still be better for `parameters_raw`
  - embedded catalog may still be better for `gguf_sources`

## Implementation Plan

### Phase 1: Installed LM Studio Overlay

Files likely touched:

- [providers.rs](/home/nullvoid/projects/llmfit/llmfit-core/src/providers.rs)
- [models.rs](/home/nullvoid/projects/llmfit/llmfit-core/src/models.rs)
- [tui_app.rs](/home/nullvoid/projects/llmfit/llmfit-tui/src/tui_app.rs)
- [serve_api.rs](/home/nullvoid/projects/llmfit/llmfit-tui/src/serve_api.rs)

Work:

1. Extend the LM Studio provider to fetch richer installed-model metadata.
2. Add a parser for whatever LM Studio exposes beyond `GET /v1/models`.
3. Build a `HashMap<String, ModelMetadataOverlay>` keyed by canonical slug.
4. Merge overlays into loaded models after `ModelDatabase::new()`.
5. Update UI/API/fit code to use effective metadata.

Acceptance criteria:

- installed LM Studio model with richer context metadata shows that value in TUI and API
- fit calculations use the effective context, not only the base HF value

### Phase 2: Online LM Studio Metadata Refresh

Files likely touched:

- [update.rs](/home/nullvoid/projects/llmfit/llmfit-core/src/update.rs)
- new LM Studio catalog scraper/helper in `llmfit-core`

Work:

1. Add a small LM Studio metadata client:
   - fetch catalog page
   - fetch model page
   - extract:
     - model slug
     - context length
     - capabilities
     - summary text
2. Run this as part of `--refresh-models`.
3. Cache LM Studio-derived overlays alongside HF refresh cache.
4. Merge at load time using field precedence.

Acceptance criteria:

- `llmfit --refresh-models` can update model context/capabilities from LM Studio when available
- refreshed LM Studio metadata survives restart via cache

### Phase 3: LM Studio as Discovery Source

Files likely touched:

- [update.rs](/home/nullvoid/projects/llmfit/llmfit-core/src/update.rs)
- [models.rs](/home/nullvoid/projects/llmfit/llmfit-core/src/models.rs)
- docs

Work:

1. Add LM Studio catalog discovery pass.
2. Map LM Studio catalog entries to canonical slugs.
3. Insert new cache-only models when not already present from HF.
4. Mark source as `LmStudioCatalog`.

Acceptance criteria:

- new models can enter the cache from LM Studio even if they were not present in the embedded HF list
- duplicates merge correctly by canonical slug

## Canonical Matching Strategy

This is a key risk area.

LM Studio names often differ from Hugging Face names by:

- publisher prefix
- variant suffix
- quant suffix
- packaging suffix
- community publisher name

We should:

1. keep using `canonical_slug(...)`
2. add provider-specific normalization helpers for LM Studio artifacts
3. support a secondary `artifact_slug(...)` for package-level matching

Examples:

- `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
- `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF`
- `lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF`

These should likely collapse to one base model slug plus multiple artifact slugs.

## Cache Design

Do not mix all sources into one untyped bucket.

Recommended:

- keep HF cache as base model cache
- add LM Studio overlay cache separately

Possible files:

- `~/.llmfit/hf_models_cache.json`
- `~/.llmfit/lmstudio_metadata_cache.json`

Why:

- easier invalidation
- easier debugging
- avoids accidental loss of source identity

## API Surface

Once implemented, expose source-awareness in API output.

Recommended additions:

- `metadata_source`
- `context_source`
- `artifact_variant`
- `installed_providers`

This will make mismatches much easier to explain.

## UI Changes

Recommended:

1. Show when displayed metadata is coming from LM Studio.
2. In detail view, show:
   - `Base Context`
   - `Effective Context`
   - `Source`
3. If a local artifact overrides the base model:
   - render a short note like `Ctx from LM Studio artifact`

This is important for trust and debuggability.

## Risks

### HTML scraping fragility

LM Studio public pages may change structure.

Mitigation:

- prefer structured local/API sources when available
- isolate LM Studio scraping in one module
- tolerate partial failure without breaking startup

### False merges

Different artifacts may be incorrectly merged into one base model.

Mitigation:

- separate base slug from artifact slug
- only apply artifact overrides to matching variants

### Overwriting better data

LM Studio may be better for some fields but worse for others.

Mitigation:

- field-level precedence, not record-level replacement

## Testing Plan

### Unit tests

- canonical slug and artifact slug normalization
- precedence resolution for `effective_context_length`
- merge behavior between:
  - embedded HF
  - refreshed HF
  - LM Studio overlay
  - local installed overlay

### Integration tests

- mock LM Studio local API responses
- refresh flow with LM Studio metadata present
- TUI/app state reload preserving selected model while metadata changes

### Regression tests

Create tests specifically for known mismatch cases:

- `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- any future model where HF and LM Studio differ on max context

## Suggested First Commit

The best first implementation is:

- no public LM Studio scraping yet
- add support for local LM Studio artifact metadata overlays
- resolve effective context length from installed artifacts

Why:

- smallest useful step
- directly addresses the user-visible mismatch
- low risk compared with catalog-wide scraping

## Suggested Follow-Up Commit

Add LM Studio online metadata cache and refresh integration.

This should extend the existing refresh flow in [update.rs](/home/nullvoid/projects/llmfit/llmfit-core/src/update.rs) rather than create a second unrelated update path.

## Non-Goals

Not part of the first pass:

- replacing Hugging Face as the main embedded catalog source
- downloading model weights just to inspect metadata
- treating every LM Studio artifact as a separate top-level model row by default

## Summary

This is feasible without a major rewrite.

The correct approach is:

1. keep Hugging Face as the base model source
2. add LM Studio as an overlay source
3. prefer local artifact metadata when it exists
4. move to field-level precedence instead of single-record truth

If implemented in that order, `llmfit` can stay stable while becoming much more accurate for LM Studio users.
