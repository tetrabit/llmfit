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

## Agreed Next Feature Branch Plan

This section captures the clarified goal for the next branch.

### User Goal

The immediate goal is not only local installed-model overlays.

The desired outcome is that `llmfit` can:

1. see LM Studio catalog models from `https://lmstudio.ai/models`
2. use LM Studio-specific metadata when it is more accurate than Hugging Face
3. download those LM Studio catalog models from the existing TUI download flow

Concrete motivating example:

- LM Studio page: `https://lmstudio.ai/models/nvidia/nemotron-3-nano`
- desired behavior: `llmfit` should surface that LM Studio model and reflect the LM Studio context window (`1,048,576`) instead of only the Hugging Face-side context metadata

### Revised Scope for the Next Branch

The next feature branch should combine:

- **Phase 2** for LM Studio online metadata refresh of known models
- the smallest useful subset of **Phase 3** for LM Studio catalog discovery

This is larger than the original local-only Phase 1, but it matches the actual requested user outcome.

### Recommended Delivery Order

#### Step 1 — Add shared overlay support in core

Before any scraping/discovery work, add the data model support needed by both local and online LM Studio metadata:

- add a `ModelMetadataOverlay` attached to `LlmModel`
- add field-level effective helpers:
  - `effective_context_length()`
  - `effective_capabilities()`
  - `effective_use_case()`
- defer `effective_format()` unless we have strong evidence it is needed immediately for runtime behavior

This keeps Hugging Face as the base model source while allowing LM Studio metadata to override selected fields safely.

#### Step 2 — Add LM Studio online metadata cache

Extend the existing refresh path in `llmfit-core/src/update.rs` with a second cache file:

- `~/.llmfit/hf_models_cache.json`
- `~/.llmfit/lmstudio_metadata_cache.json`

The LM Studio cache should store overlays, not full replacement models.

For known models already present in the embedded/HF model list:

- fetch LM Studio page/catalog metadata
- extract fields like:
  - canonical key / slug
  - context length
  - capabilities
  - display name / summary
  - artifact or variant identifiers where available
- save those as overlays keyed by canonical slug

#### Step 3 — Apply overlays in the shared load path

Do not apply LM Studio metadata only in the TUI.

Create a shared model-loading path so:

- TUI
- API
- CLI

all see the same effective model metadata.

This avoids a split-brain state where the TUI shows LM Studio metadata but `serve_api` and CLI still show Hugging Face-only values.

#### Step 4 — Add catalog discovery for LM Studio-only entries

After known-model overlays are working, add the smallest discovery pass that can insert new models from LM Studio when they are not already present from Hugging Face.

Requirements:

- dedupe by canonical slug
- avoid record-level replacement of HF entries
- insert new cache-only entries only when there is no existing base model
- mark discovered entries as LM Studio-derived

This is the step that makes “LM Studio-specific models” actually appear in `llmfit`.

#### Step 5 — Keep download flow aligned with LM Studio keys

Ensure discovered/overlaid LM Studio models retain enough information to use the correct LM Studio download identifier.

This is critical because some LM Studio catalog models use LM Studio-specific keys or mappings that differ from raw Hugging Face repo IDs.

### Initial Acceptance Criteria for the Branch

The next branch should be considered successful when all of the following are true:

1. After refreshing models, `llmfit` can surface LM Studio metadata for known models like `nvidia/nemotron-3-nano`
2. The Nemotron example reflects the LM Studio context window (`1,048,576`) in fit/UI/API output
3. New LM Studio catalog entries can appear in the model set even when they are absent from the embedded HF catalog
4. Downloading an LM Studio-discovered model uses the correct LM Studio identifier or mapping
5. LM Studio metadata precedence remains field-level, not whole-record replacement

### Known Risks to Handle Explicitly

1. **Fragile scraping**
   - LM Studio public pages may change
   - prefer structured data embedded in the page when available
   - isolate parsing in one helper/module

2. **False slug merges**
   - one LM Studio artifact can map ambiguously onto a Hugging Face base model
   - if ambiguous, skip overlay rather than overwrite incorrectly

3. **Download identifier mismatch**
   - the correct LM Studio download key may differ from the visible Hugging Face repo ID
   - persist the LM Studio key explicitly when known

4. **Cross-surface inconsistency**
   - TUI/API/CLI must all read effective metadata from the same shared load path

### Regression Cases to Preserve

At minimum, add branch tests or fixtures covering:

- `nvidia/nemotron-3-nano`
- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- a model where LM Studio capabilities differ from the HF-inferred capabilities
- a model discovered from LM Studio but not present in the embedded HF catalog

### Branch Strategy

This work should start on a dedicated feature branch after the current local LM Studio/runtime/filter fixes are committed and pushed.

Recommended branch intent:

- isolate the larger LM Studio metadata/catalog work from the already-validated runtime/download fixes
- allow incremental PRs if the branch naturally splits into:
  1. core overlay model support
  2. LM Studio online metadata cache
  3. LM Studio catalog discovery
