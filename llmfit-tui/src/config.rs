use llmfit_core::fit::EstimationContextMode;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(default)]
pub struct AppConfig {
    pub estimation_context_mode: EstimationContextMode,
}

impl AppConfig {
    pub fn config_path() -> Option<PathBuf> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()?;
        Some(
            PathBuf::from(home)
                .join(".config")
                .join("llmfit")
                .join("config.json"),
        )
    }

    pub fn load() -> Option<Self> {
        Self::config_path().and_then(|path| Self::load_from_path(&path))
    }

    pub fn load_from_path(path: &Path) -> Option<Self> {
        let content = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::AppConfig;
    use llmfit_core::fit::EstimationContextMode;

    fn temp_config_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "llmfit-config-test-{}-{}-{}.json",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        );
        path.push(unique);
        path
    }

    #[test]
    fn config_loads_default_from_empty_json() {
        let path = temp_config_path("default");
        std::fs::write(&path, "{}\n").expect("write config");

        let loaded = AppConfig::load_from_path(&path).expect("load config");
        assert_eq!(
            loaded.estimation_context_mode,
            EstimationContextMode::DefaultCapped
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn config_roundtrips_model_max_mode() {
        let path = temp_config_path("model-max");
        let config = AppConfig {
            estimation_context_mode: EstimationContextMode::ModelMax,
        };

        std::fs::write(
            &path,
            serde_json::to_string_pretty(&config).expect("serialize config"),
        )
        .expect("write config");
        let loaded = AppConfig::load_from_path(&path).expect("load config");
        assert_eq!(loaded, config);

        let _ = std::fs::remove_file(path);
    }
}
