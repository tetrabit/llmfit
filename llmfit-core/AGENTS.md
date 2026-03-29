# llmfit-core KNOWLEDGE BASE

## OVERVIEW

Shared engine crate: hardware detection, model parsing, fit/scoring, provider integration, plan estimates, and online cache updates.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Rank/fit/runtime choice | `src/fit.rs` | Biggest logic file; fit levels, quant selection, speed estimates |
| Hardware probing | `src/hardware.rs` | NVIDIA/AMD/Intel/Apple/Ascend + cluster/unified memory |
| Embedded model DB | `src/models.rs`, `data/hf_models.json` | `include_str!("../data/hf_models.json")` |
| Planning mode math | `src/plan.rs` | Hardware requirement estimation |
| Provider runtime integration | `src/providers.rs` | Pull/install detection and provider mapping tables |
| Online updates/cache | `src/update.rs` | Hugging Face API fetch + local cache merge |

## CONVENTIONS

- `src/lib.rs` is the public surface; keep exports intentional.
- `ModelDatabase::new()` merges embedded models with cached online updates.
- Distinguish hardware backend (`GpuBackend`) from inference runtime (`InferenceRuntime`).
- Pre-quantized or cluster-mode flows can force `vLLM`; do not assume `llama.cpp` everywhere.
- Provider code is best-effort around network/process availability; preserve graceful failure behavior.

## ANTI-PATTERNS

- Editing embedded JSON by hand instead of regenerating it.
- Breaking `FitLevel` / `RunMode` ordering without updating ranking logic.
- Collapsing unified-memory, MoE offload, CPU offload, and tensor-parallel paths into one generic branch.
- Treating provider installed-model probes as exact truth without normalization/deduping.

## PACKAGE GOTCHAS

- `providers.rs`, `fit.rs`, and `hardware.rs` are all large; search first, then patch surgically.
- `scripts/scrape_docker_models.py` and `scripts/scrape_hf_models.py` feed `llmfit-core/data/*`.
- This crate has many inline tests near file bottoms; prefer extending those before inventing new test layouts.

## COMMANDS

```bash
cargo test -p llmfit-core
cargo check -p llmfit-core
python3 scripts/scrape_hf_models.py
python3 scripts/scrape_docker_models.py
```
