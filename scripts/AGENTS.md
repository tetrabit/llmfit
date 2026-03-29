# scripts KNOWLEDGE BASE

## OVERVIEW

Repo maintenance scripts for regenerating model data, validating API behavior, and installing the bundled OpenClaw skill.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| HF model regeneration | `scrape_hf_models.py` | Writes both root and `llmfit-core/data` HF JSON |
| Docker model mapping regen | `scrape_docker_models.py` | Writes `llmfit-core/data/docker_models.json` |
| End-to-end API smoke tests | `test_api.py` | Can spawn `cargo run -p llmfit -- serve` |
| Remote model verification | `verify_models.py` | HF path is current; Ollama path still reflects old single-crate layout |
| Legacy update wrapper | `update_models.sh` | Still talks about root `data/hf_models.json` |
| Skill install | `install-openclaw-skill.sh` | Copies `skills/llmfit-advisor` into OpenClaw |

## CONVENTIONS

- Prefer Python stdlib/network scripts; avoid adding pip dependencies casually.
- Treat generated JSON as script output, not hand-maintained source.
- After regeneration, verify the Rust build still embeds the updated package-local data.

## ANTI-PATTERNS

- Updating only root `data/hf_models.json` and forgetting `llmfit-core/data/hf_models.json`.
- Assuming script comments still match workspace reality without checking code paths.
- Running verification scripts from the wrong working directory; several resolve paths from repo root.

## PACKAGE GOTCHAS

- `update_models.sh` is conservative but slightly stale in wording; actual scraper mirrors files into both locations.
- `verify_models.py --ollama` still points at the old root-level providers path; verify or fix that before relying on it.
- `test_api.py --spawn` is the fastest way to validate the served JSON envelope after backend changes.

## COMMANDS

```bash
python3 scripts/scrape_hf_models.py
python3 scripts/scrape_docker_models.py
python3 scripts/test_api.py --spawn
python3 scripts/verify_models.py --hf
```
