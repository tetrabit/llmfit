//! Online model database updates via the HuggingFace Hub API.
//!
//! Fetches trending / top-downloaded text-generation models and caches them in
//! `~/.llmfit/hf_models_cache.json` (Linux/macOS) or
//! `%APPDATA%\llmfit\hf_models_cache.json` (Windows).
//!
//! The cache is automatically merged with the embedded model list each time
//! `ModelDatabase::new()` is called, so users immediately benefit from any
//! previously fetched models without needing to rebuild the binary.

use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::models::{Capability, LlmModel, MetadataSource, ModelFormat, ModelMetadataOverlay};

const HF_API: &str = "https://huggingface.co/api/models";
const LMSTUDIO_MODELS_URL: &str = "https://lmstudio.ai/models";

/// Bump this when the `LlmModel` schema changes in a breaking way.
/// A cache written by an older version will be discarded and re-fetched.
const CACHE_VERSION: u32 = 1;

// ── Cache helpers ─────────────────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct CacheEnvelope {
    version: u32,
    models: Vec<LlmModel>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct OverlayCacheEnvelope {
    version: u32,
    overlays: HashMap<String, ModelMetadataOverlay>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LmStudioModelPageMetadata {
    model_key: String,
    context_length: Option<u32>,
    capabilities: Vec<Capability>,
    parameter_count: Option<String>,
    artifact_name: Option<String>,
    notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
struct LmStudioCatalogEntry {
    model_key: String,
    memory_gb: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct LmStudioCatalogFamilyPage {
    concrete_models: Vec<LmStudioCatalogEntry>,
}

/// Returns the llmfit data directory.
/// `~/.llmfit` on Linux/macOS, `%APPDATA%\llmfit` on Windows.
pub fn cache_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("APPDATA")
            .ok()
            .map(|p| PathBuf::from(p).join("llmfit"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()
            .map(|p| PathBuf::from(p).join(".llmfit"))
    }
}

/// Full path to the cached model list JSON file.
pub fn cache_file() -> Option<PathBuf> {
    Some(cache_dir()?.join("hf_models_cache.json"))
}

pub fn lmstudio_metadata_cache_file() -> Option<PathBuf> {
    Some(cache_dir()?.join("lmstudio_metadata_cache.json"))
}

/// Load any previously cached models.
///
/// Returns an empty vec if the cache is missing, corrupt, or was written by
/// a different schema version (triggering a silent re-fetch on next update).
pub fn load_cache() -> Vec<LlmModel> {
    let path = match cache_file() {
        Some(p) if p.exists() => p,
        _ => return vec![],
    };
    let Ok(content) = std::fs::read_to_string(&path) else {
        return vec![];
    };
    match serde_json::from_str::<CacheEnvelope>(&content) {
        Ok(env) if env.version == CACHE_VERSION => env.models,
        // Version mismatch or old unversioned format — discard stale cache.
        _ => vec![],
    }
}

pub fn load_lmstudio_metadata_cache() -> HashMap<String, ModelMetadataOverlay> {
    let path = match lmstudio_metadata_cache_file() {
        Some(p) if p.exists() => p,
        _ => return HashMap::new(),
    };
    let Ok(content) = std::fs::read_to_string(&path) else {
        return HashMap::new();
    };
    match serde_json::from_str::<OverlayCacheEnvelope>(&content) {
        Ok(env) if env.version == CACHE_VERSION => env.overlays,
        _ => HashMap::new(),
    }
}

/// Persist a model list to the cache file, creating the directory if needed.
pub fn save_cache(models: &[LlmModel]) -> Result<(), String> {
    let path = cache_file().ok_or_else(|| "Cannot determine cache directory".to_string())?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("Failed to create cache dir: {e}"))?;
    }
    let envelope = CacheEnvelope {
        version: CACHE_VERSION,
        models: models.to_vec(),
    };
    let json =
        serde_json::to_string_pretty(&envelope).map_err(|e| format!("Serialize error: {e}"))?;
    std::fs::write(&path, json).map_err(|e| format!("Failed to write cache: {e}"))?;
    Ok(())
}

pub fn save_lmstudio_metadata_cache(
    overlays: &HashMap<String, ModelMetadataOverlay>,
) -> Result<(), String> {
    let path = lmstudio_metadata_cache_file()
        .ok_or_else(|| "Cannot determine cache directory".to_string())?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("Failed to create cache dir: {e}"))?;
    }
    let envelope = OverlayCacheEnvelope {
        version: CACHE_VERSION,
        overlays: overlays.clone(),
    };
    let json =
        serde_json::to_string_pretty(&envelope).map_err(|e| format!("Serialize error: {e}"))?;
    std::fs::write(&path, json).map_err(|e| format!("Failed to write cache: {e}"))?;
    Ok(())
}

/// Delete the cache file.  Returns the number of models that were removed.
pub fn clear_cache() -> Result<usize, String> {
    let path = match cache_file() {
        Some(p) if p.exists() => p,
        _ => return Ok(0),
    };
    let count = load_cache().len();
    std::fs::remove_file(&path).map_err(|e| format!("Failed to delete cache: {e}"))?;
    Ok(count)
}

pub fn clear_lmstudio_metadata_cache() -> Result<usize, String> {
    let path = match lmstudio_metadata_cache_file() {
        Some(p) if p.exists() => p,
        _ => return Ok(0),
    };
    let count = load_lmstudio_metadata_cache().len();
    std::fs::remove_file(&path).map_err(|e| format!("Failed to delete cache: {e}"))?;
    Ok(count)
}

// ── HuggingFace API types ─────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct HfApiModel {
    id: String,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    pipeline_tag: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(rename = "createdAt", default)]
    created_at: Option<String>,
    /// Exact parameter count when safetensors metadata is available.
    #[serde(default)]
    safetensors: Option<SafetensorsInfo>,
    #[serde(default)]
    license: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct SafetensorsInfo {
    #[serde(default)]
    total: Option<u64>,
}

// ── Parameter extraction ──────────────────────────────────────────────────────

/// Parse "7B" → 7_000_000_000u64, "1.5B" → 1_500_000_000u64, "500M" → 500_000_000u64.
fn parse_param_str(s: &str) -> Option<u64> {
    if let Some(b) = s.strip_suffix('B') {
        let v: f64 = b.parse().ok()?;
        if (0.05..=2_000_000.0).contains(&v) {
            return Some((v * 1_000_000_000.0) as u64);
        }
    }
    if let Some(m) = s.strip_suffix('M') {
        let v: f64 = m.parse().ok()?;
        if (1.0..=999_999.0).contains(&v) {
            return Some((v * 1_000_000.0) as u64);
        }
    }
    None
}

/// Parse "16E", "128E" → Some(16), Some(128).
fn parse_expert_suffix(s: &str) -> Option<u32> {
    s.strip_suffix('E')?.parse().ok()
}

/// Derive (param_str, params_raw, is_moe, num_experts, active_experts, active_params)
/// from a model identifier string.
fn extract_model_params(
    model_id: &str,
) -> (
    String,
    Option<u64>,
    bool,
    Option<u32>,
    Option<u32>,
    Option<u64>,
) {
    let up = model_id.to_uppercase();
    // Split on typical separators but NOT on '.' so that "1.5B" stays intact.
    let tokens: Vec<&str> = up.split(['-', '/', '_', ' ', ':']).collect();

    // MoE pattern "8X7B": N experts × M params each.
    for tok in &tokens {
        if let Some(x_pos) = tok.find('X') {
            let (left, right) = tok.split_at(x_pos);
            let right = &right[1..]; // skip 'X'
            if let (Ok(n_exp), Some(per_exp)) = (left.parse::<u32>(), parse_param_str(right))
                && (2..=512).contains(&n_exp)
            {
                let total = per_exp.saturating_mul(n_exp as u64);
                let active_exp = 2u32.min(n_exp);
                let active = per_exp.saturating_mul(active_exp as u64);
                let pb = per_exp / 1_000_000_000;
                let s = if pb > 0 {
                    format!("{}x{}B", n_exp, pb)
                } else {
                    format!("{}x{}M", n_exp, per_exp / 1_000_000)
                };
                return (
                    s,
                    Some(total),
                    true,
                    Some(n_exp),
                    Some(active_exp),
                    Some(active),
                );
            }
        }
    }

    // MoE pattern "17B-16E": per-expert params + expert count.
    for window in tokens.windows(2) {
        if let (Some(pb), Some(ne)) = (parse_param_str(window[0]), parse_expert_suffix(window[1]))
            && (2..=512).contains(&ne)
        {
            let total = pb.saturating_mul(ne as u64);
            let ae = 2u32.min(ne);
            let active = pb.saturating_mul(ae as u64);
            let s = format!("{}B", pb / 1_000_000_000);
            return (s, Some(total), true, Some(ne), Some(ae), Some(active));
        }
    }

    // Standard dense model: first matching NB / NM token wins.
    for tok in &tokens {
        if let Some(raw) = parse_param_str(tok) {
            let b = raw / 1_000_000_000;
            let frac = (raw % 1_000_000_000) / 100_000_000;
            let s = if b > 0 {
                if frac > 0 {
                    format!("{}.{}B", b, frac)
                } else {
                    format!("{}B", b)
                }
            } else {
                format!("{}M", raw / 1_000_000)
            };
            return (s, Some(raw), false, None, None, None);
        }
    }

    // Could not parse param count from the model name — mark as unknown rather
    // than silently guessing 7B, which would produce misleading RAM estimates.
    ("Unknown".to_string(), None, false, None, None, None)
}

// ── Use-case inference ────────────────────────────────────────────────────────

fn infer_use_case(model_id: &str, tags: &[String]) -> String {
    let lower = format!(
        "{} {}",
        model_id.to_lowercase(),
        tags.join(" ").to_lowercase()
    );
    if lower.contains("embed") || lower.contains("bge") || lower.contains("-e5-") {
        "Embedding".to_string()
    } else if lower.contains("tool")
        || lower.contains("function call")
        || lower.contains("agent")
        || lower.contains("qwen3")
        || lower.contains("qwen2.5")
        || lower.contains("command-r")
        || lower.contains("hermes")
    {
        "Agentic & tool use".to_string()
    } else if lower.contains("code") || lower.contains("starcoder") || lower.contains("coder") {
        "Code generation".to_string()
    } else if lower.contains("vision")
        || lower.contains("-vl")
        || lower.contains("llava")
        || lower.contains("multimodal")
    {
        "Vision & Language".to_string()
    } else if lower.contains("-r1")
        || lower.contains("reasoning")
        || lower.contains("thinking")
        || lower.contains("qwq")
    {
        "Reasoning & chain-of-thought".to_string()
    } else if lower.contains("instruct") || lower.contains("chat") || lower.contains("assistant") {
        "Chat & instruction following".to_string()
    } else {
        "General text generation".to_string()
    }
}

// ── Context-length inference ──────────────────────────────────────────────────

fn apply_context_family_floor(model_id: &str, context_length: u32) -> u32 {
    let low = model_id.to_lowercase();

    if low.contains("llama-4") || low.contains("llama4") {
        return context_length.max(1_048_576);
    }
    if low.contains("nemotron-3-nano-30b-a3b") {
        return context_length.max(1_048_576);
    }
    if low.contains("llama-3") || low.contains("llama3") {
        return context_length.max(131_072);
    }
    if low.contains("qwen2") || low.contains("qwen3") {
        return context_length.max(131_072);
    }
    if low.contains("mistral") {
        return context_length.max(32_768);
    }
    if low.contains("gemma") {
        return context_length.max(8_192);
    }
    context_length
}

fn infer_context_length(model_id: &str, params_raw: Option<u64>) -> u32 {
    let low = model_id.to_lowercase();
    for (kw, ctx) in &[
        ("1m", 1_048_576u32),
        ("128k", 131_072),
        ("64k", 65_536),
        ("32k", 32_768),
        ("16k", 16_384),
        ("8k", 8_192),
    ] {
        if low.contains(kw) {
            return apply_context_family_floor(model_id, *ctx);
        }
    }

    match params_raw {
        Some(p) if p >= 70_000_000_000 => apply_context_family_floor(model_id, 32_768),
        Some(p) if p >= 13_000_000_000 => apply_context_family_floor(model_id, 16_384),
        Some(p) if p >= 3_000_000_000 => apply_context_family_floor(model_id, 8_192),
        _ => apply_context_family_floor(model_id, 4_096),
    }
}

fn exact_context_length(model_id: &str, config: &Value) -> Option<u32> {
    let keys = [
        "max_position_embeddings",
        "max_sequence_length",
        "model_max_length",
        "seq_length",
        "n_positions",
        "sliding_window",
    ];

    for key in keys {
        if let Some(value) = config.get(key).and_then(Value::as_u64)
            && let Ok(value) = u32::try_from(value)
            && value > 0
        {
            return Some(apply_context_family_floor(model_id, value));
        }
    }

    if let Some(text_config) = config.get("text_config").and_then(Value::as_object) {
        for key in [
            "max_position_embeddings",
            "max_sequence_length",
            "model_max_length",
            "seq_length",
            "n_positions",
            "sliding_window",
        ] {
            if let Some(value) = text_config.get(key).and_then(Value::as_u64)
                && let Ok(value) = u32::try_from(value)
                && value > 0
            {
                return Some(apply_context_family_floor(model_id, value));
            }
        }
    }

    None
}

// ── RAM estimation ────────────────────────────────────────────────────────────

fn estimate_ram(
    params_raw: u64,
    is_moe: bool,
    active_params: Option<u64>,
) -> (f64, f64, Option<f64>) {
    let total_b = params_raw as f64 / 1_000_000_000.0;
    let active_b = active_params
        .map(|a| a as f64 / 1_000_000_000.0)
        .unwrap_or(total_b);
    let gpu_b = if is_moe { active_b } else { total_b };
    // Q2_K bpp ≈ 0.37 → absolute floor
    let min_ram = (total_b * 0.37 + 0.5).max(1.0);
    // Q4_K_M bpp ≈ 0.58 → comfortable default
    let rec_ram = (total_b * 0.58 + 1.0).max(2.0);
    let min_vram = (gpu_b * 0.58 + 0.5).max(1.0);
    (min_ram, rec_ram, Some(min_vram))
}

// ── HF API fetching ───────────────────────────────────────────────────────────

fn hf_get_list(sort: &str, limit: usize, token: Option<&str>) -> Result<Vec<HfApiModel>, String> {
    let url = format!(
        "{}?pipeline_tag=text-generation&sort={}&limit={}",
        HF_API, sort, limit
    );
    let resp = if let Some(t) = token {
        ureq::get(&url)
            .header("Authorization", &format!("Bearer {}", t))
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(30)))
            .build()
            .call()
    } else {
        ureq::get(&url)
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(30)))
            .build()
            .call()
    };
    match resp {
        Ok(r) => r
            .into_body()
            .read_json::<Vec<HfApiModel>>()
            .map_err(|e| format!("Failed to parse HuggingFace API response: {e}")),
        Err(e) => {
            let msg = e.to_string();
            Err(if msg.contains("401") || msg.contains("Unauthorized") {
                "HTTP 401 Unauthorized — is HF_TOKEN set and valid?".to_string()
            } else if msg.contains("403") || msg.contains("Forbidden") {
                "HTTP 403 Forbidden — token may lack read permission".to_string()
            } else if msg.contains("429") || msg.contains("Too Many") {
                "HTTP 429 Rate limited — wait a moment and retry".to_string()
            } else {
                format!("HuggingFace API error: {e}")
            })
        }
    }
}

fn hf_get_model(repo_id: &str, token: Option<&str>) -> Result<HfApiModel, String> {
    let url = format!("{}/{}", HF_API, repo_id);
    let resp = if let Some(t) = token {
        ureq::get(&url)
            .header("Authorization", &format!("Bearer {}", t))
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(30)))
            .build()
            .call()
    } else {
        ureq::get(&url)
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(30)))
            .build()
            .call()
    };
    match resp {
        Ok(r) => r
            .into_body()
            .read_json::<HfApiModel>()
            .map_err(|e| format!("Failed to parse HuggingFace model response: {e}")),
        Err(e) => Err(format!(
            "HuggingFace model lookup failed for {repo_id}: {e}"
        )),
    }
}

fn fetch_config_json(repo_id: &str, token: Option<&str>) -> Option<Value> {
    let url = format!("https://huggingface.co/{repo_id}/resolve/main/config.json");
    let resp = if let Some(t) = token {
        ureq::get(&url)
            .header("Authorization", &format!("Bearer {}", t))
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(30)))
            .build()
            .call()
    } else {
        ureq::get(&url)
            .config()
            .timeout_global(Some(std::time::Duration::from_secs(30)))
            .build()
            .call()
    };
    let Ok(resp) = resp else {
        return None;
    };
    resp.into_body().read_json::<Value>().ok()
}

fn fetch_precise_context_length(repo_id: &str, token: Option<&str>) -> Option<u32> {
    fetch_config_json(repo_id, token).and_then(|config| exact_context_length(repo_id, &config))
}

fn lmstudio_get(url: &str) -> Result<String, String> {
    ureq::get(url)
        .header("User-Agent", "Mozilla/5.0")
        .config()
        .timeout_global(Some(std::time::Duration::from_secs(30)))
        .build()
        .call()
        .map_err(|e| format!("LM Studio request failed for {url}: {e}"))?
        .into_body()
        .read_to_string()
        .map_err(|e| format!("Failed to read LM Studio response for {url}: {e}"))
}

fn lmstudio_model_url(model_key: &str) -> String {
    format!("{}/{}", LMSTUDIO_MODELS_URL, model_key)
}

fn decode_html_text(input: &str) -> String {
    input
        .replace("\\u003c", "<")
        .replace("\\u003e", ">")
        .replace("\\u0026", "&")
        .replace("\\\"", "\"")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&amp;", "&")
}

fn strip_html_tags(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut in_tag = false;
    for ch in input.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out
}

fn normalize_space(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn parse_lmstudio_catalog_index(html: &str) -> Vec<String> {
    let mut slugs = Vec::new();
    let marker = "href=\"/models/";
    let mut start = 0;

    while let Some(idx) = html[start..].find(marker) {
        let href_start = start + idx + marker.len();
        let Some(end_rel) = html[href_start..].find('"') else {
            break;
        };
        let slug = &html[href_start..href_start + end_rel];
        start = href_start + end_rel;

        if slug.is_empty() || slug.contains('/') {
            continue;
        }
        if !slugs.iter().any(|existing| existing == slug) {
            slugs.push(slug.to_string());
        }
    }

    slugs
}

fn parse_lmstudio_family_page(html: &str) -> LmStudioCatalogFamilyPage {
    let mut concrete_models = Vec::new();
    let mut seen = HashSet::new();

    if let Some(keywords_start) = html.find("\"keywords\":[") {
        let tail = &html[keywords_start + "\"keywords\": [".len() - 1..];
        if let Some(end_rel) = tail.find(']') {
            let keywords_block = &tail[..end_rel];
            for keyword in keywords_block.split(',') {
                let keyword = keyword.trim().trim_matches('"');
                if let Some((owner, model)) = keyword.split_once('/')
                    && !owner.is_empty()
                    && !model.is_empty()
                    && !model.contains(' ')
                    && seen.insert(keyword.to_string())
                {
                    concrete_models.push(LmStudioCatalogEntry {
                        model_key: keyword.to_string(),
                        memory_gb: None,
                    });
                }
            }
        }
    }

    let marker = "href=\"/models/";
    let mut start = 0;

    while let Some(idx) = html[start..].find(marker) {
        let href_start = start + idx + marker.len();
        let Some(end_rel) = html[href_start..].find('"') else {
            break;
        };
        let slug = &html[href_start..href_start + end_rel];
        start = href_start + end_rel;

        let Some((owner, model)) = slug.split_once('/') else {
            continue;
        };
        if owner.is_empty() || model.is_empty() {
            continue;
        }

        let window_end = (start + 400).min(html.len());
        let window = &html[start..window_end];
        let memory_gb = extract_first_memory_gb(window);
        let model_key = format!("{owner}/{model}");

        if seen.insert(model_key.clone()) {
            concrete_models.push(LmStudioCatalogEntry {
                model_key,
                memory_gb,
            });
        }
    }

    LmStudioCatalogFamilyPage { concrete_models }
}

fn extract_first_memory_gb(input: &str) -> Option<f64> {
    let mut digits = String::new();
    let mut seen_digit = false;

    for ch in input.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
            seen_digit = true;
        } else if ch == '.' && seen_digit && !digits.contains('.') {
            digits.push(ch);
        } else if seen_digit {
            let rest = &input[input.find(&digits)? + digits.len()..];
            if rest.trim_start().starts_with("GB") {
                return digits.parse::<f64>().ok();
            }
            digits.clear();
            seen_digit = false;
        }
    }

    None
}

fn parse_lmstudio_model_page(html: &str) -> Option<LmStudioModelPageMetadata> {
    if let Some(parsed) = parse_lmstudio_model_page_from_yaml(html) {
        return Some(parsed);
    }
    parse_lmstudio_model_page_from_content(html)
}

fn parse_lmstudio_model_page_from_yaml(html: &str) -> Option<LmStudioModelPageMetadata> {
    let title_start = html.find("# model.yaml is an open standard")?;
    let title_end = html[title_start..]
        .find("</pre>")
        .map(|idx| title_start + idx)?;
    let block = decode_html_text(&html[title_start..title_end]);
    let text = strip_html_tags(&block);
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect::<Vec<_>>();

    let mut model_key = None;
    let mut context_length = None;
    let mut parameter_count = None;
    let mut artifact_name = None;
    let mut vision = None;
    let mut tool_use = None;
    let mut first_base_key = None;

    for window in lines.windows(2) {
        let line = window[0];
        if let Some(value) = line.strip_prefix("model:") {
            model_key = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("key:") {
            let key = value.trim().to_string();
            if first_base_key.is_none() && key.starts_with("lmstudio-community/") {
                first_base_key = Some(key);
            }
        } else if let Some(value) = line.strip_prefix("repo:") {
            if artifact_name.is_none() {
                artifact_name = Some(value.trim().to_string());
            }
        } else if let Some(value) = line.strip_prefix("vision:") {
            vision = Some(value.trim().eq_ignore_ascii_case("true"));
        } else if let Some(value) = line.strip_prefix("trainedForToolUse:") {
            tool_use = Some(value.trim().eq_ignore_ascii_case("true"));
        } else if line.starts_with("paramsStrings:") {
            let next = window[1].trim_start_matches('-').trim();
            if !next.contains(':') {
                parameter_count = Some(next.to_string());
            }
        } else if line.starts_with("contextLengths:") {
            let next = window[1].trim_start_matches('-').trim();
            if !next.contains(':') {
                context_length = next.parse::<u32>().ok();
            }
        }
    }

    if let Some(last_line) = lines.last()
        && let Some(value) = last_line.strip_prefix("trainedForToolUse:")
    {
        tool_use = Some(value.trim().eq_ignore_ascii_case("true"));
    }

    let model_key = model_key?;
    let mut capabilities = Vec::new();
    if vision.unwrap_or(false) {
        capabilities.push(Capability::Vision);
    }
    if tool_use.unwrap_or(false) {
        capabilities.push(Capability::ToolUse);
    }

    Some(LmStudioModelPageMetadata {
        model_key,
        context_length,
        capabilities,
        parameter_count,
        artifact_name: artifact_name.or(first_base_key),
        notes: Some("LM Studio catalog metadata".to_string()),
    })
}

fn parse_lmstudio_model_page_from_content(html: &str) -> Option<LmStudioModelPageMetadata> {
    let decoded = decode_html_text(html);
    let text = normalize_space(&strip_html_tags(&decoded));

    let model_key = extract_model_key_from_page(&decoded, &text)?;
    let artifact_name = extract_lmstudio_artifact_repo(&decoded);
    let context_length = extract_context_length_from_text(&text);
    let parameter_count = extract_parameter_count_from_text(&text);

    let lower = text.to_lowercase();
    let mut capabilities = Vec::new();
    if lower.contains("tool use")
        || lower.contains("tool calling")
        || lower.contains("function calling")
        || lower.contains("agentic")
        || lower.contains("web browsing")
        || lower.contains("python execution")
    {
        capabilities.push(Capability::ToolUse);
    }
    if lower.contains("vision") || lower.contains("multimodal") {
        capabilities.push(Capability::Vision);
    }

    Some(LmStudioModelPageMetadata {
        model_key,
        context_length,
        capabilities,
        parameter_count,
        artifact_name,
        notes: Some("LM Studio catalog metadata".to_string()),
    })
}

fn extract_model_key_from_page(decoded_html: &str, text: &str) -> Option<String> {
    if let Some(pos) = decoded_html.find("\"keywords\":[") {
        let tail = &decoded_html[pos..];
        if let Some(end) = tail.find(']') {
            let block = &tail[..end];
            for keyword in block.split(',') {
                let keyword = keyword.trim().trim_matches('"');
                if let Some((owner, model)) = keyword.split_once('/')
                    && !owner.is_empty()
                    && !model.is_empty()
                    && !model.contains(' ')
                    && !model.chars().any(|ch| ch.is_uppercase())
                {
                    return Some(keyword.to_string());
                }
            }
        }
    }

    let marker = "← All Models";
    if let Some(idx) = text.find(marker) {
        let after = text[idx + marker.len()..].trim_start();
        let first = after.split_whitespace().next()?;
        if first.contains('/') {
            return Some(first.to_string());
        }
    }

    None
}

fn extract_lmstudio_artifact_repo(decoded_html: &str) -> Option<String> {
    let marker = "huggingface.co/lmstudio-community/";
    let start = decoded_html.find(marker)? + "huggingface.co/".len();
    let tail = &decoded_html[start..];
    let end = tail
        .find(|c: char| c == '"' || c == '<' || c == '>' || c.is_whitespace())
        .unwrap_or(tail.len());
    Some(tail[..end].trim_end_matches('/').to_string())
}

fn extract_context_length_from_text(text: &str) -> Option<u32> {
    let lower = text.to_lowercase();
    if lower.contains("1m tokens") || lower.contains("1 million tokens") {
        return Some(1_048_576);
    }
    if lower.contains("256k") {
        return Some(262_144);
    }
    if lower.contains("131k") || lower.contains("128k") {
        return Some(131_072);
    }
    None
}

fn extract_parameter_count_from_text(text: &str) -> Option<String> {
    let words = text.split_whitespace().collect::<Vec<_>>();
    for idx in 0..words.len() {
        let token = words[idx].trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '.');
        if !(token.ends_with('B') || token.ends_with('M')) {
            continue;
        }
        let prefix = &token[..token.len().saturating_sub(1)];
        if !(prefix.chars().all(|ch| ch.is_ascii_digit() || ch == '.') && !prefix.is_empty()) {
            continue;
        }

        let next = words
            .get(idx + 1)
            .map(|w| w.trim_matches(|c: char| !c.is_ascii_alphabetic()))
            .unwrap_or("")
            .to_lowercase();
        let prev = words
            .get(idx.saturating_sub(1))
            .map(|w| w.trim_matches(|c: char| !c.is_ascii_alphabetic()))
            .unwrap_or("")
            .to_lowercase();

        if matches!(next.as_str(), "total" | "parameters" | "parameter")
            || matches!(prev.as_str(), "with" | "contains" | "has")
        {
            return Some(token.to_string());
        }
    }

    for token in &words {
        let token = token.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '.');
        if token.ends_with('B') {
            let prefix = &token[..token.len().saturating_sub(1)];
            if prefix.chars().all(|ch| ch.is_ascii_digit() || ch == '.') && !prefix.is_empty() {
                return Some(token.to_string());
            }
        }
    }
    None
}

fn fetch_lmstudio_model_page(model_key: &str) -> Result<Option<LmStudioModelPageMetadata>, String> {
    let html = lmstudio_get(&lmstudio_model_url(model_key))?;
    Ok(parse_lmstudio_model_page(&html))
}

fn fetch_lmstudio_overlay_for_model(
    model_key: &str,
) -> Result<Option<ModelMetadataOverlay>, String> {
    let Some(page) = fetch_lmstudio_model_page(model_key)? else {
        return Ok(None);
    };

    let use_case = if page.capabilities.contains(&Capability::ToolUse) {
        Some("Agentic & tool use".to_string())
    } else {
        None
    };

    Ok(Some(ModelMetadataOverlay {
        source: MetadataSource::LmStudioCatalog,
        context_length: page.context_length,
        capabilities: if page.capabilities.is_empty() {
            None
        } else {
            Some(page.capabilities)
        },
        use_case,
        artifact_name: page.artifact_name,
        notes: page.notes,
    }))
}

fn resolve_lmstudio_overlay_for_model(
    repo_id: &str,
) -> Result<Option<ModelMetadataOverlay>, String> {
    let candidates = crate::providers::hf_name_to_lmstudio_candidates(repo_id);
    let mut last_error = None;

    for candidate in candidates {
        match fetch_lmstudio_overlay_for_model(&candidate) {
            Ok(Some(overlay)) => return Ok(Some(overlay)),
            Ok(None) => {}
            Err(err) => {
                if !err.contains("404") {
                    last_error = Some(err);
                }
            }
        }
    }

    if let Some(err) = last_error {
        Err(err)
    } else {
        Ok(None)
    }
}

fn build_lmstudio_catalog_model(
    model_key: &str,
    page: &LmStudioModelPageMetadata,
    memory_gb_hint: Option<f64>,
) -> Option<LlmModel> {
    let parameter_count = page
        .parameter_count
        .clone()
        .unwrap_or_else(|| "Unknown".to_string());
    let (parsed_label, params_raw, is_moe, num_experts, active_experts, active_parameters) =
        extract_model_params(model_key);
    let params_raw =
        params_raw.or_else(|| page.parameter_count.as_deref().and_then(parse_param_str));
    let parameter_count = if page.parameter_count.is_some() {
        parameter_count
    } else {
        parsed_label
    };

    let raw = params_raw?;
    let context_length = page
        .context_length
        .unwrap_or_else(|| infer_context_length(model_key, params_raw));
    let use_case = if page.capabilities.contains(&Capability::ToolUse) {
        "Agentic & tool use".to_string()
    } else if page.capabilities.contains(&Capability::Vision) {
        "Vision & Language".to_string()
    } else {
        infer_use_case(model_key, &[])
    };

    let (mut min_ram_gb, mut recommended_ram_gb, min_vram_gb) =
        estimate_ram(raw, is_moe, active_parameters);
    if let Some(memory_gb) = memory_gb_hint {
        min_ram_gb = min_ram_gb.max(memory_gb);
        recommended_ram_gb = recommended_ram_gb.max(memory_gb);
    }

    let provider = model_key
        .split('/')
        .next()
        .unwrap_or("LM Studio")
        .to_string();
    let artifact_name = page
        .artifact_name
        .clone()
        .unwrap_or_else(|| model_key.to_string());

    Some(LlmModel {
        name: model_key.to_string(),
        provider,
        parameter_count,
        parameters_raw: params_raw,
        min_ram_gb,
        recommended_ram_gb,
        min_vram_gb,
        quantization: "Q4_K_M".to_string(),
        context_length,
        use_case,
        is_moe,
        num_experts,
        active_experts,
        active_parameters,
        release_date: None,
        gguf_sources: vec![crate::models::GgufSource {
            repo: artifact_name.clone(),
            provider: artifact_name
                .split('/')
                .next()
                .unwrap_or("lmstudio-community")
                .to_string(),
        }],
        capabilities: page.capabilities.clone(),
        format: ModelFormat::Gguf,
        num_attention_heads: None,
        num_key_value_heads: None,
        metadata_overlay: Some(ModelMetadataOverlay {
            source: MetadataSource::LmStudioCatalog,
            context_length: page.context_length,
            capabilities: (!page.capabilities.is_empty()).then_some(page.capabilities.clone()),
            use_case: Some(if page.capabilities.contains(&Capability::ToolUse) {
                "Agentic & tool use".to_string()
            } else if page.capabilities.contains(&Capability::Vision) {
                "Vision & Language".to_string()
            } else {
                infer_use_case(model_key, &[])
            }),
            artifact_name: page.artifact_name.clone(),
            notes: page.notes.clone(),
        }),
        license: None,
    })
}

fn fetch_lmstudio_catalog_models(progress: impl Fn(&str)) -> Result<Vec<LlmModel>, String> {
    let index_html = lmstudio_get(LMSTUDIO_MODELS_URL)?;
    let family_slugs = parse_lmstudio_catalog_index(&index_html);
    let mut models = Vec::new();
    let mut seen = HashSet::new();

    for (family_idx, family_slug) in family_slugs.iter().enumerate() {
        let family_html = match lmstudio_get(&lmstudio_model_url(family_slug)) {
            Ok(html) => html,
            Err(err) => {
                progress(&format!(
                    "  Warning: LM Studio family fetch failed for {family_slug} — {err}"
                ));
                continue;
            }
        };
        let family_page = parse_lmstudio_family_page(&family_html);

        for entry in family_page.concrete_models {
            if !seen.insert(entry.model_key.clone()) {
                continue;
            }

            match fetch_lmstudio_model_page(&entry.model_key) {
                Ok(Some(page)) => {
                    if let Some(model) =
                        build_lmstudio_catalog_model(&entry.model_key, &page, entry.memory_gb)
                    {
                        models.push(model);
                    }
                }
                Ok(None) => {}
                Err(err) => progress(&format!(
                    "  Warning: LM Studio model fetch failed for {} — {}",
                    entry.model_key, err
                )),
            }
        }

        if (family_idx + 1) % 10 == 0 || family_idx + 1 == family_slugs.len() {
            progress(&format!(
                "  Crawled {}/{} LM Studio catalog families",
                family_idx + 1,
                family_slugs.len()
            ));
        }
    }

    Ok(models)
}

/// Convert a raw HF API entry into an `LlmModel`.
/// Returns `None` for models that cannot be characterised as text-generation.
/// Detect the model format from HF API tags and model name.
///
/// The HF API includes library tags like `"gguf"`, `"safetensors"`,
/// `"transformers"` as well as quantization hints in the model name
/// (e.g. `-AWQ`, `-GPTQ`, `-MLX`).  We check tags first (authoritative),
/// then fall back to name-based heuristics.
fn detect_format_from_hf(model_id: &str, tags: &[String]) -> (ModelFormat, String) {
    let has_tag = |t: &str| tags.iter().any(|tag| tag.eq_ignore_ascii_case(t));
    let name_upper = model_id.to_uppercase();

    // AWQ — tag or name
    if has_tag("awq") || name_upper.contains("-AWQ") {
        let bits = if name_upper.contains("8BIT") {
            "8bit"
        } else {
            "4bit"
        };
        return (ModelFormat::Awq, format!("AWQ-{bits}"));
    }

    // GPTQ — tag or name
    if has_tag("gptq") || name_upper.contains("-GPTQ") {
        let bits = if name_upper.contains("INT8") || name_upper.contains("8BIT") {
            "Int8"
        } else {
            "Int4"
        };
        return (ModelFormat::Gptq, format!("GPTQ-{bits}"));
    }

    // MLX — tag or name
    if has_tag("mlx") || name_upper.contains("-MLX") {
        return (ModelFormat::Mlx, "MLX".to_string());
    }

    // Explicit GGUF tag or -GGUF in name → GGUF
    if has_tag("gguf") || name_upper.contains("-GGUF") {
        return (ModelFormat::Gguf, "Q4_K_M".to_string());
    }

    // safetensors tag (without any quant tag) → Safetensors
    if has_tag("safetensors") || has_tag("transformers") {
        return (ModelFormat::Safetensors, "BF16".to_string());
    }

    // No information → default to Safetensors (conservative: don't claim GGUF
    // when we have no evidence of GGUF files).
    (ModelFormat::Safetensors, "BF16".to_string())
}

fn map_to_llm_model(hf: HfApiModel, precise_context_length: Option<u32>) -> Option<LlmModel> {
    let is_tg = hf.pipeline_tag.as_deref() == Some("text-generation")
        || hf.tags.iter().any(|t| t == "text-generation");
    if !is_tg {
        return None;
    }

    // Use safetensors for an exact parameter count when available, but always
    // run name-based parsing for MoE architecture hints — safetensors only
    // reports total parameters and would cause MoE models (e.g. Mixtral) to
    // lose their MoE classification and receive inaccurate VRAM estimates.
    let (param_str, params_raw, is_moe, num_experts, active_experts, active_params) =
        if let Some(total) = hf.safetensors.as_ref().and_then(|s| s.total) {
            let (_, _, is_moe, num_experts, active_experts, active_params) =
                extract_model_params(&hf.id);
            let b = total / 1_000_000_000;
            let frac = (total % 1_000_000_000) / 100_000_000;
            let s = if b > 0 {
                if frac > 0 {
                    format!("{}.{}B", b, frac)
                } else {
                    format!("{}B", b)
                }
            } else {
                format!("{}M", total / 1_000_000)
            };
            (
                s,
                Some(total),
                is_moe,
                num_experts,
                active_experts,
                active_params,
            )
        } else {
            extract_model_params(&hf.id)
        };

    let raw = params_raw.unwrap_or(7_000_000_000);
    let use_case = infer_use_case(&hf.id, &hf.tags);
    let context_length =
        precise_context_length.unwrap_or_else(|| infer_context_length(&hf.id, params_raw));
    let (min_ram, rec_ram, min_vram) = estimate_ram(raw, is_moe, active_params);
    let (format, quantization) = detect_format_from_hf(&hf.id, &hf.tags);

    let provider = hf
        .author
        .as_deref()
        .or_else(|| hf.id.split('/').next())
        .unwrap_or("Unknown")
        .to_string();

    let release_date = hf
        .created_at
        .as_deref()
        .map(|s| s.get(..10).unwrap_or(s).to_string());

    let license = hf.license.or_else(|| {
        hf.tags
            .iter()
            .find_map(|t| t.strip_prefix("license:").map(|l| l.to_string()))
    });

    Some(LlmModel {
        name: hf.id,
        provider,
        parameter_count: param_str,
        parameters_raw: params_raw,
        min_ram_gb: min_ram,
        recommended_ram_gb: rec_ram,
        min_vram_gb: min_vram,
        quantization,
        context_length,
        use_case,
        is_moe,
        num_experts,
        active_experts,
        active_parameters: active_params,
        release_date,
        gguf_sources: vec![],
        capabilities: vec![],
        format,
        num_attention_heads: None,
        num_key_value_heads: None,
        metadata_overlay: None,
        license,
    })
}

fn fetch_model_metadata(repo_id: &str, token: Option<&str>) -> Result<Option<LlmModel>, String> {
    let hf = hf_get_model(repo_id, token)?;
    let precise_context_length = fetch_precise_context_length(repo_id, token);
    Ok(map_to_llm_model(hf, precise_context_length))
}

fn upsert_cached_model(cache: &mut HashMap<String, LlmModel>, model: LlmModel) -> bool {
    let slug = crate::models::canonical_slug(&model.name);
    match cache.get(&slug) {
        Some(existing) if existing == &model => false,
        _ => {
            cache.insert(slug, model);
            true
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Options controlling the update behaviour.
#[derive(Debug, Clone)]
pub struct UpdateOptions {
    /// Number of trending models to fetch (0 = skip).
    pub trending_limit: usize,
    /// Number of top-downloaded models to fetch (0 = skip).
    pub downloads_limit: usize,
    /// Optional HuggingFace API token (raises the anonymous rate limit).
    pub token: Option<String>,
    /// Re-query known catalog models so refreshed metadata can override the
    /// embedded catalog, not just append new discoveries.
    pub refresh_existing: bool,
    /// Optional explicit set of model repo IDs to refresh. When non-empty,
    /// only these known models are re-queried.
    pub specific_models: Vec<String>,
}

impl Default for UpdateOptions {
    fn default() -> Self {
        Self {
            trending_limit: 100,
            downloads_limit: 50,
            token: None,
            refresh_existing: false,
            specific_models: Vec::new(),
        }
    }
}

/// Fetch new models from HuggingFace and merge them into the local cache.
///
/// Returns `Ok((new_count, total_cached))` on success, where `new_count` is
/// the number of models added in this run and `total_cached` is the size of the
/// cache file after the update.
///
/// `progress` receives human-readable status strings suitable for printing to
/// stdout or displaying in a TUI.
pub fn update_model_cache(
    opts: &UpdateOptions,
    progress: impl Fn(&str),
) -> Result<(usize, usize), String> {
    use crate::models::ModelDatabase;

    let mut cached_by_slug: HashMap<String, LlmModel> = load_cache()
        .into_iter()
        .map(|m| (crate::models::canonical_slug(&m.name), m))
        .collect();
    let mut lmstudio_overlays = load_lmstudio_metadata_cache();

    let token = opts.token.as_deref();
    let mut all_hf: Vec<HfApiModel> = Vec::new();
    let mut changed_count = 0usize;
    let mut fetched_any = false;

    if opts.refresh_existing {
        let mut known_model_ids: Vec<String> = if opts.specific_models.is_empty() {
            ModelDatabase::new()
                .get_all_models()
                .iter()
                .map(|m| m.name.clone())
                .filter(|name| name.contains('/'))
                .collect::<HashSet<_>>()
                .into_iter()
                .collect()
        } else {
            opts.specific_models
                .iter()
                .filter(|name| name.contains('/'))
                .cloned()
                .collect::<HashSet<_>>()
                .into_iter()
                .collect()
        };
        known_model_ids.sort();

        progress(&format!(
            "Refreshing {} known models from HuggingFace...",
            known_model_ids.len()
        ));

        for (idx, repo_id) in known_model_ids.iter().enumerate() {
            match fetch_model_metadata(repo_id, token) {
                Ok(Some(model)) => {
                    fetched_any = true;
                    if upsert_cached_model(&mut cached_by_slug, model) {
                        changed_count += 1;
                    }
                }
                Ok(None) => {}
                Err(e) => progress(&format!("  Warning: refresh failed for {repo_id} — {e}")),
            }

            match resolve_lmstudio_overlay_for_model(repo_id) {
                Ok(Some(overlay)) => {
                    fetched_any = true;
                    let slug = crate::models::canonical_slug(repo_id);
                    if lmstudio_overlays.get(&slug) != Some(&overlay) {
                        lmstudio_overlays.insert(slug, overlay);
                    }
                }
                Ok(None) => {}
                Err(e) => progress(&format!(
                    "  Warning: LM Studio metadata failed for {repo_id} — {e}"
                )),
            }

            if (idx + 1) % 25 == 0 || idx + 1 == known_model_ids.len() {
                progress(&format!(
                    "  Refreshed {}/{} known models",
                    idx + 1,
                    known_model_ids.len()
                ));
            }
        }
    }

    if opts.trending_limit > 0 {
        progress(&format!(
            "Fetching {} trending models from HuggingFace...",
            opts.trending_limit
        ));
        match hf_get_list("trendingScore", opts.trending_limit, token) {
            Ok(list) => {
                fetched_any = true;
                progress(&format!("  Received {} trending models", list.len()));
                all_hf.extend(list);
            }
            Err(e) => progress(&format!("  Warning: trending fetch failed — {e}")),
        }
    }

    if opts.downloads_limit > 0 {
        progress(&format!(
            "Fetching {} top-downloaded models...",
            opts.downloads_limit
        ));
        match hf_get_list("downloads", opts.downloads_limit, token) {
            Ok(list) => {
                fetched_any = true;
                progress(&format!("  Received {} download-ranked models", list.len()));
                all_hf.extend(list);
            }
            Err(e) => progress(&format!("  Warning: downloads fetch failed — {e}")),
        }
    }

    progress("Fetching LM Studio catalog models...");
    match fetch_lmstudio_catalog_models(&progress) {
        Ok(lmstudio_models) => {
            if !lmstudio_models.is_empty() {
                fetched_any = true;
            }
            progress(&format!(
                "  Received {} LM Studio catalog models",
                lmstudio_models.len()
            ));
            for model in lmstudio_models {
                if upsert_cached_model(&mut cached_by_slug, model) {
                    changed_count += 1;
                }
            }
        }
        Err(e) => progress(&format!("  Warning: LM Studio catalog fetch failed — {e}")),
    }

    if !fetched_any {
        return Err("No models fetched — check your internet connection".to_string());
    }

    // Deduplicate by ID (trending and downloads lists can overlap).
    let mut seen: HashSet<String> = HashSet::new();
    all_hf.retain(|m| seen.insert(m.id.clone()));

    progress(&format!("Processing {} unique models...", all_hf.len()));

    for hf in all_hf {
        let precise_context_length = fetch_precise_context_length(&hf.id, token);
        if let Some(model) = map_to_llm_model(hf, precise_context_length)
            && upsert_cached_model(&mut cached_by_slug, model)
        {
            changed_count += 1;
        }
    }

    let mut cached: Vec<LlmModel> = cached_by_slug.into_values().collect();
    cached.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    let total = cached.len();
    progress(&format!(
        "Saving {} cached models ({} updated or added)...",
        total, changed_count
    ));
    save_cache(&cached)?;
    save_lmstudio_metadata_cache(&lmstudio_overlays)?;

    Ok((changed_count, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MetadataSource;

    #[test]
    fn test_parse_param_str_billions() {
        assert_eq!(parse_param_str("7B"), Some(7_000_000_000));
        assert_eq!(parse_param_str("70B"), Some(70_000_000_000));
        assert_eq!(parse_param_str("1.5B"), Some(1_500_000_000));
        assert_eq!(parse_param_str("405B"), Some(405_000_000_000));
    }

    #[test]
    fn test_parse_param_str_millions() {
        assert_eq!(parse_param_str("500M"), Some(500_000_000));
        assert_eq!(parse_param_str("135M"), Some(135_000_000));
    }

    #[test]
    fn test_parse_param_str_invalid() {
        assert_eq!(parse_param_str("INSTRUCT"), None);
        assert_eq!(parse_param_str(""), None);
        assert_eq!(parse_param_str("0.001B"), None); // below 0.05 threshold
    }

    #[test]
    fn test_extract_dense_model() {
        let (s, raw, moe, ne, ae, ap) = extract_model_params("meta-llama/Llama-3.1-8B-Instruct");
        assert_eq!(s, "8B");
        assert_eq!(raw, Some(8_000_000_000));
        assert!(!moe);
        assert!(ne.is_none());
        assert!(ae.is_none());
        assert!(ap.is_none());
    }

    #[test]
    fn test_extract_moe_nx_model() {
        let (s, raw, moe, ne, ae, _ap) = extract_model_params("mistralai/Mixtral-8x7B-v0.1");
        assert!(s.contains("8x"), "Expected MoE label, got: {}", s);
        assert!(raw.is_some());
        assert!(moe);
        assert_eq!(ne, Some(8));
        assert_eq!(ae, Some(2));
    }

    #[test]
    fn test_extract_fractional_param() {
        let (s, raw, ..) = extract_model_params("Qwen/Qwen2.5-1.5B-Instruct");
        assert!(s.contains("1.5") || s.contains("1"), "got: {}", s);
        assert!(raw.is_some());
    }

    #[test]
    fn test_infer_use_case_coding() {
        let uc = infer_use_case("deepseek-ai/DeepSeek-Coder-6.7B", &[]);
        assert!(uc.to_lowercase().contains("code"), "got: {}", uc);
    }

    #[test]
    fn test_infer_use_case_embedding() {
        let uc = infer_use_case("BAAI/bge-large-en-v1.5", &[]);
        assert!(uc.to_lowercase().contains("embed"), "got: {}", uc);
    }

    #[test]
    fn test_infer_use_case_agentic() {
        let uc = infer_use_case(
            "Qwen/Qwen3-8B",
            &["tool-use".to_string(), "agents".to_string()],
        );
        assert!(uc.to_lowercase().contains("agent"), "got: {}", uc);
    }

    #[test]
    fn test_infer_context_length_keywords() {
        assert_eq!(infer_context_length("model-128k", None), 131_072);
        assert_eq!(infer_context_length("model-32k", None), 32_768);
    }

    #[test]
    fn test_infer_context_length_llama3() {
        // Llama-3 family defaults to 128 k context
        assert_eq!(
            infer_context_length("meta-llama/Llama-3.1-8B", None),
            131_072
        );
    }

    #[test]
    fn test_infer_context_length_nemotron_nano() {
        assert_eq!(
            infer_context_length("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", None),
            1_048_576
        );
    }

    #[test]
    fn test_estimate_ram_dense() {
        let (min_r, rec_r, vram) = estimate_ram(7_000_000_000, false, None);
        assert!(min_r > 2.0 && min_r < 5.0, "min_ram={}", min_r);
        assert!(rec_r > min_r, "rec_ram should exceed min_ram");
        assert!(vram.is_some());
    }

    #[test]
    fn test_estimate_ram_moe() {
        // MoE: total=56B, active=14B → VRAM based on active only
        let (_, _, vram_moe) = estimate_ram(56_000_000_000, true, Some(14_000_000_000));
        let (_, _, vram_dense) = estimate_ram(56_000_000_000, false, None);
        // MoE VRAM should be substantially lower than dense equivalent
        assert!(vram_moe.unwrap() < vram_dense.unwrap());
    }

    #[test]
    fn test_parse_lmstudio_model_page_nemotron_metadata() {
        let html = r#"
        <pre><code>
        # model.yaml is an open standard for defining cross-platform, composable AI models
        model: nvidia/nemotron-3-nano
        base:
          - key: lmstudio-community/nvidia-nemotron-3-nano-30b-a3b-gguf
            sources:
              - type: huggingface
                user: lmstudio-community
                repo: NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF
        metadataOverrides:
          domain: llm
          architectures:
            - nemotron_h_moe
          compatibilityTypes:
            - gguf
            - safetensors
          paramsStrings:
            - 30B
          minMemoryUsageBytes: 24620000000
          contextLengths:
            - 1048576
          vision: false
          reasoning: true
          trainedForToolUse: true
        </code></pre>
        "#;

        let parsed = parse_lmstudio_model_page(html).expect("should parse lmstudio model page");

        assert_eq!(parsed.model_key, "nvidia/nemotron-3-nano");
        assert_eq!(parsed.context_length, Some(1_048_576));
        assert_eq!(parsed.parameter_count.as_deref(), Some("30B"));
        assert!(parsed.capabilities.contains(&Capability::ToolUse));
        assert!(!parsed.capabilities.contains(&Capability::Vision));
        assert_eq!(
            parsed.artifact_name.as_deref(),
            Some("NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF")
        );
    }

    #[test]
    fn test_fetch_lmstudio_overlay_maps_metadata_to_overlay() {
        let html = r#"
        <pre><code>
        # model.yaml is an open standard for defining cross-platform, composable AI models
        model: nvidia/nemotron-3-nano
        metadataOverrides:
          compatibilityTypes:
            - gguf
          paramsStrings:
            - 30B
          contextLengths:
            - 1048576
          vision: false
          trainedForToolUse: true
        </code></pre>
        "#;

        let parsed = parse_lmstudio_model_page(html).expect("should parse model yaml block");
        let overlay = ModelMetadataOverlay {
            source: MetadataSource::LmStudioCatalog,
            context_length: parsed.context_length,
            capabilities: Some(parsed.capabilities),
            use_case: Some("Agentic & tool use".to_string()),
            artifact_name: parsed.artifact_name,
            notes: parsed.notes,
        };

        assert_eq!(overlay.source, MetadataSource::LmStudioCatalog);
        assert_eq!(overlay.context_length, Some(1_048_576));
        assert_eq!(overlay.use_case.as_deref(), Some("Agentic & tool use"));
        assert!(
            overlay
                .capabilities
                .as_ref()
                .expect("caps present")
                .contains(&Capability::ToolUse)
        );
    }

    #[test]
    fn test_lmstudio_candidate_mapping_includes_nemotron_public_slug() {
        let candidates = crate::providers::hf_name_to_lmstudio_candidates(
            "stelterlab/NVIDIA-Nemotron-3-Nano-30B-A3B-AWQ",
        );

        assert!(candidates.contains(&"nvidia/nemotron-3-nano".to_string()));
    }

    #[test]
    fn test_parse_lmstudio_catalog_index_extracts_family_slugs() {
        let html = r#"
        <a href="/models/qwen3">Qwen3</a>
        <a href="/models/nemotron-3">Nemotron 3</a>
        <a href="/models/qwen/qwen3-4b-2507">Concrete</a>
        "#;

        assert_eq!(
            parse_lmstudio_catalog_index(html),
            vec!["qwen3".to_string(), "nemotron-3".to_string()]
        );
    }

    #[test]
    fn test_parse_lmstudio_family_page_extracts_concrete_models() {
        let html = r#"
        <a href="/models/qwen/qwen3-coder-30b">qwen/qwen3-coder-30b</a>
        <div>15.00 GB</div>
        <a href="/models/openai/gpt-oss-20b">openai/gpt-oss-20b</a>
        <div>12.00 GB</div>
        "#;

        let parsed = parse_lmstudio_family_page(html);

        assert_eq!(
            parsed.concrete_models,
            vec![
                LmStudioCatalogEntry {
                    model_key: "qwen/qwen3-coder-30b".to_string(),
                    memory_gb: Some(15.0),
                },
                LmStudioCatalogEntry {
                    model_key: "openai/gpt-oss-20b".to_string(),
                    memory_gb: Some(12.0),
                }
            ]
        );
    }

    #[test]
    fn test_parse_lmstudio_family_page_extracts_keywords_models() {
        let html = r#"
        <script type="application/ld+json">
        {"@type":"CreativeWork","keywords":["Qwen3","qwen/qwen3-4b-2507","qwen/qwen3-coder-30b"]}
        </script>
        "#;

        let parsed = parse_lmstudio_family_page(html);

        assert_eq!(
            parsed.concrete_models,
            vec![
                LmStudioCatalogEntry {
                    model_key: "qwen/qwen3-4b-2507".to_string(),
                    memory_gb: None,
                },
                LmStudioCatalogEntry {
                    model_key: "qwen/qwen3-coder-30b".to_string(),
                    memory_gb: None,
                }
            ]
        );
    }

    #[test]
    fn test_parse_lmstudio_model_page_falls_back_to_narrative_content() {
        let html = r#"
        <script type="application/ld+json">
        {"@type":"CreativeWork","keywords":["Nemotron 3","nvidia/nemotron-3-nano"]}
        </script>
        <a href="/models/nvidia/nemotron-3-nano">nvidia/nemotron-3-nano</a>
        <p>Supports a context length of 1M tokens.</p>
        <p>General purpose reasoning and chat model trained from scratch by NVIDIA. Contains 30B total parameters with only 3.5B active at a time for low-latency MoE inference.</p>
        <p>Nemotron 3 models support tool use and reasoning. They are available in gguf and mlx.</p>
        <a href="https://huggingface.co/lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF">artifact</a>
        "#;

        let parsed = parse_lmstudio_model_page(html).expect("should parse fallback content");

        assert_eq!(parsed.model_key, "nvidia/nemotron-3-nano");
        assert_eq!(parsed.context_length, Some(1_048_576));
        assert_eq!(parsed.parameter_count.as_deref(), Some("30B"));
        assert!(parsed.capabilities.contains(&Capability::ToolUse));
        assert_eq!(
            parsed.artifact_name.as_deref(),
            Some("lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF")
        );
    }

    #[test]
    fn test_build_lmstudio_catalog_model_maps_page_to_llm_model() {
        let page = LmStudioModelPageMetadata {
            model_key: "qwen/qwen3-coder-30b".to_string(),
            context_length: Some(262_144),
            capabilities: vec![Capability::ToolUse],
            parameter_count: Some("30B".to_string()),
            artifact_name: Some("lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF".to_string()),
            notes: Some("LM Studio catalog metadata".to_string()),
        };

        let model = build_lmstudio_catalog_model("qwen/qwen3-coder-30b", &page, Some(15.0))
            .expect("model should build");

        assert_eq!(model.name, "qwen/qwen3-coder-30b");
        assert_eq!(model.provider, "qwen");
        assert_eq!(model.parameter_count, "30B");
        assert_eq!(model.context_length, 262_144);
        assert!(model.min_ram_gb >= 15.0);
        assert_eq!(model.metadata_source(), MetadataSource::LmStudioCatalog);
        assert!(
            model
                .effective_capabilities()
                .contains(&Capability::ToolUse)
        );
        assert_eq!(
            model.gguf_sources[0].repo,
            "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF"
        );
    }

    #[test]
    fn test_build_lmstudio_catalog_model_uses_page_parameter_count_when_slug_lacks_size() {
        let page = LmStudioModelPageMetadata {
            model_key: "nvidia/nemotron-3-nano".to_string(),
            context_length: Some(1_048_576),
            capabilities: vec![Capability::ToolUse],
            parameter_count: Some("30B".to_string()),
            artifact_name: Some(
                "lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-GGUF".to_string(),
            ),
            notes: Some("LM Studio catalog metadata".to_string()),
        };

        let model = build_lmstudio_catalog_model("nvidia/nemotron-3-nano", &page, Some(25.0))
            .expect("model should build from page parameter count");

        assert_eq!(model.name, "nvidia/nemotron-3-nano");
        assert_eq!(model.parameter_count, "30B");
        assert_eq!(model.parameters_raw, Some(30_000_000_000));
        assert_eq!(model.effective_context_length(), 1_048_576);
    }

    #[test]
    fn test_catalog_only_model_survives_cache_round_trip_into_model_database() {
        let cache_path = cache_file().expect("cache path");
        let overlay_path = lmstudio_metadata_cache_file().expect("overlay path");
        let original_cache = std::fs::read_to_string(&cache_path).ok();
        let original_overlay = std::fs::read_to_string(&overlay_path).ok();

        let page = LmStudioModelPageMetadata {
            model_key: "openai/gpt-oss-20b".to_string(),
            context_length: Some(131_072),
            capabilities: vec![Capability::ToolUse],
            parameter_count: Some("20B".to_string()),
            artifact_name: Some("lmstudio-community/gpt-oss-20b-gguf".to_string()),
            notes: Some("LM Studio catalog metadata".to_string()),
        };

        let model = build_lmstudio_catalog_model("openai/gpt-oss-20b", &page, Some(12.0))
            .expect("catalog model should build");
        save_cache(&[model]).expect("save cache");

        let db = crate::models::ModelDatabase::new();
        let imported = db
            .get_all_models()
            .iter()
            .find(|model| model.name == "openai/gpt-oss-20b")
            .expect("imported model present");

        assert_eq!(imported.provider, "openai");
        assert_eq!(imported.effective_context_length(), 131_072);
        assert!(
            imported
                .effective_capabilities()
                .contains(&Capability::ToolUse)
        );

        if let Some(content) = original_cache {
            if let Some(dir) = cache_path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(&cache_path, content);
        } else {
            let _ = std::fs::remove_file(&cache_path);
        }
        if let Some(content) = original_overlay {
            if let Some(dir) = overlay_path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(&overlay_path, content);
        } else {
            let _ = std::fs::remove_file(&overlay_path);
        }
    }
}
