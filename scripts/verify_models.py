#!/usr/bin/env python3
"""Verify that all models in hf_models.json exist on HuggingFace and all
Ollama mappings in src/providers.rs point to valid Ollama registry entries.

Usage:
    python3 scripts/verify_models.py            # check both
    python3 scripts/verify_models.py --hf       # HuggingFace only
    python3 scripts/verify_models.py --ollama   # Ollama only

Exits with code 1 if any model is missing. Suitable for CI.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_MODELS_PATH = REPO_ROOT / "data" / "hf_models.json"
EMBEDDED_HF_MODELS_PATH = REPO_ROOT / "llmfit-core" / "data" / "hf_models.json"
PROVIDERS_RS_PATH = REPO_ROOT / "llmfit-core" / "src" / "providers.rs"

HEADERS = {"User-Agent": "llmfit-verify/1.0"}
REQUEST_DELAY = 0.3  # seconds between requests to avoid rate limiting


def check_url(url: str) -> int:
    """GET a URL and return the HTTP status code, or -1 on error."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        resp = urllib.request.urlopen(req, timeout=10)
        return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# HuggingFace verification
# ---------------------------------------------------------------------------


def load_hf_models() -> list[str]:
    """Return list of HF repo names from hf_models.json."""
    with open(HF_MODELS_PATH) as f:
        data = json.load(f)
    return [m["name"] for m in data]


def verify_hf(models: list[str]) -> list[str]:
    """Check each HF model exists. Returns list of missing model names."""
    missing = []
    total = len(models)
    for i, name in enumerate(models, 1):
        url = f"https://huggingface.co/api/models/{name}"
        status = check_url(url)
        if status == 200:
            print(f"  [{i}/{total}] ✓ {name}")
        else:
            print(f"  [{i}/{total}] ✗ {name} (HTTP {status})")
            missing.append(name)
        time.sleep(REQUEST_DELAY)
    return missing


# ---------------------------------------------------------------------------
# Ollama verification
# ---------------------------------------------------------------------------


def parse_ollama_tags() -> list[str]:
    """Extract unique Ollama tags from OLLAMA_MAPPINGS in providers.rs."""
    src = PROVIDERS_RS_PATH.read_text()

    # Find the OLLAMA_MAPPINGS block
    match = re.search(
        r"const OLLAMA_MAPPINGS:.*?=.*?\[(.+?)\];",
        src,
        re.DOTALL,
    )
    if not match:
        print("ERROR: Could not find OLLAMA_MAPPINGS in providers.rs")
        sys.exit(2)

    block = match.group(1)
    # Extract the second element of each tuple: ("hf_name", "ollama_tag")
    tags = re.findall(r'\(\s*"[^"]+"\s*,\s*"([^"]+)"\s*\)', block)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique


def verify_ollama(tags: list[str]) -> list[str]:
    """Check each Ollama tag exists. Returns list of missing tags."""
    missing = []
    total = len(tags)
    for i, tag in enumerate(tags, 1):
        url = f"https://ollama.com/library/{tag}"
        status = check_url(url)
        if status == 200:
            print(f"  [{i}/{total}] ✓ {tag}")
        else:
            print(f"  [{i}/{total}] ✗ {tag} (HTTP {status})")
            missing.append(tag)
        time.sleep(REQUEST_DELAY)
    return missing


# ---------------------------------------------------------------------------
# Mirrored HF JSON sync verification
# ---------------------------------------------------------------------------


def load_json_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def verify_hf_sync() -> list[str]:
    failures = []
    root_bytes = load_json_bytes(HF_MODELS_PATH)
    embedded_bytes = load_json_bytes(EMBEDDED_HF_MODELS_PATH)

    root_hash = hashlib.sha256(root_bytes).hexdigest()
    embedded_hash = hashlib.sha256(embedded_bytes).hexdigest()
    root_count = len(json.loads(root_bytes))
    embedded_count = len(json.loads(embedded_bytes))

    print("\n=== HF JSON mirror sync ===\n")
    print(f"  root:     {HF_MODELS_PATH} ({root_count} models, sha256 {root_hash})")
    print(
        f"  embedded: {EMBEDDED_HF_MODELS_PATH} ({embedded_count} models, sha256 {embedded_hash})"
    )

    if root_count != embedded_count:
        failures.append(
            f"model count mismatch: root={root_count}, embedded={embedded_count}"
        )

    if root_hash != embedded_hash:
        failures.append("sha256 mismatch between root and embedded HF JSON")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Verify model availability")
    parser.add_argument("--hf", action="store_true", help="Check HuggingFace only")
    parser.add_argument("--ollama", action="store_true", help="Check Ollama only")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Check root and embedded HF JSON stay mirrored",
    )
    args = parser.parse_args()

    # Default: check both
    check_hf = args.hf or (not args.ollama and not args.sync)
    check_ollama = args.ollama or (not args.hf and not args.sync)
    check_sync = args.sync

    failures = False

    if check_hf:
        models = load_hf_models()
        print(f"\n=== HuggingFace: checking {len(models)} models ===\n")
        missing = verify_hf(models)
        if missing:
            failures = True
            print(f"\n  ⚠ {len(missing)} HuggingFace model(s) not found:")
            for m in missing:
                print(f"    - {m}")
        else:
            print(f"\n  All {len(models)} HuggingFace models verified ✓")

    if check_ollama:
        tags = parse_ollama_tags()
        print(f"\n=== Ollama: checking {len(tags)} tags ===\n")
        missing = verify_ollama(tags)
        if missing:
            failures = True
            print(f"\n  ⚠ {len(missing)} Ollama tag(s) not found:")
            for t in missing:
                print(f"    - {t}")
        else:
            print(f"\n  All {len(tags)} Ollama tags verified ✓")

    if check_sync:
        sync_failures = verify_hf_sync()
        if sync_failures:
            failures = True
            print("\n  ⚠ HF JSON mirror mismatch detected:")
            for failure in sync_failures:
                print(f"    - {failure}")
        else:
            print("\n  Root and embedded HF JSON are in sync ✓")

    print()
    if failures:
        print("FAIL: Some models are unavailable. Fix mappings or remove entries.")
        sys.exit(1)
    else:
        print("PASS: All models verified.")


if __name__ == "__main__":
    main()
