use llmfit_core::fit::{FitLevel, InferenceRuntime, ModelFit, SortColumn, backend_compatible};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::{Capability, LlmModel, ModelDatabase, UseCase};
use llmfit_core::plan::{PlanEstimate, PlanRequest, estimate_model_plan};
use llmfit_core::providers::{
    self, DockerModelRunnerProvider, LlamaCppProvider, LmStudioProvider, MlxProvider,
    ModelProvider, OllamaProvider, PullEvent, PullHandle, VllmProvider,
};
use llmfit_core::update::{self, UpdateOptions};
use serde::{Deserialize, Serialize};

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use crate::theme::Theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    Visual,
    Select,
    Search,
    Plan,
    ProviderPopup,
    UseCasePopup,
    CapabilityPopup,
    DownloadProviderPopup,
    QuantPopup,
    RunModePopup,
    ParamsBucketPopup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanField {
    Context,
    Quant,
    TargetTps,
}

impl PlanField {
    fn next(self) -> Self {
        match self {
            PlanField::Context => PlanField::Quant,
            PlanField::Quant => PlanField::TargetTps,
            PlanField::TargetTps => PlanField::Context,
        }
    }

    fn prev(self) -> Self {
        match self {
            PlanField::Context => PlanField::TargetTps,
            PlanField::Quant => PlanField::Context,
            PlanField::TargetTps => PlanField::Quant,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitFilter {
    All,
    Perfect,
    Good,
    Marginal,
    TooTight,
    Runnable, // Perfect + Good + Marginal (excludes TooTight)
}

impl FitFilter {
    pub fn label(&self) -> &str {
        match self {
            FitFilter::All => "All",
            FitFilter::Perfect => "Perfect",
            FitFilter::Good => "Good",
            FitFilter::Marginal => "Marginal",
            FitFilter::TooTight => "Too Tight",
            FitFilter::Runnable => "Runnable",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            FitFilter::All => FitFilter::Runnable,
            FitFilter::Runnable => FitFilter::Perfect,
            FitFilter::Perfect => FitFilter::Good,
            FitFilter::Good => FitFilter::Marginal,
            FitFilter::Marginal => FitFilter::TooTight,
            FitFilter::TooTight => FitFilter::All,
        }
    }

    fn from_label(label: &str) -> Self {
        match label {
            "Perfect" => FitFilter::Perfect,
            "Good" => FitFilter::Good,
            "Marginal" => FitFilter::Marginal,
            "Too Tight" => FitFilter::TooTight,
            "Runnable" => FitFilter::Runnable,
            _ => FitFilter::All,
        }
    }
}

/// Filter by model availability / download readiness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvailabilityFilter {
    All,
    HasGguf,   // Has GGUF download sources (unsloth, bartowski, etc.)
    Installed, // Already installed in a local runtime
}

impl AvailabilityFilter {
    pub fn label(&self) -> &str {
        match self {
            AvailabilityFilter::All => "All",
            AvailabilityFilter::HasGguf => "GGUF Avail",
            AvailabilityFilter::Installed => "Installed",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            AvailabilityFilter::All => AvailabilityFilter::HasGguf,
            AvailabilityFilter::HasGguf => AvailabilityFilter::Installed,
            AvailabilityFilter::Installed => AvailabilityFilter::All,
        }
    }

    fn from_label(label: &str) -> Self {
        match label {
            "GGUF Avail" => AvailabilityFilter::HasGguf,
            "Installed" => AvailabilityFilter::Installed,
            _ => AvailabilityFilter::All,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpFilter {
    All,
    Tp2,
    Tp3,
    Tp4,
}

impl TpFilter {
    pub fn label(&self) -> &str {
        match self {
            TpFilter::All => "All",
            TpFilter::Tp2 => "TP=2",
            TpFilter::Tp3 => "TP=3",
            TpFilter::Tp4 => "TP=4",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            TpFilter::All => TpFilter::Tp2,
            TpFilter::Tp2 => TpFilter::Tp3,
            TpFilter::Tp3 => TpFilter::Tp4,
            TpFilter::Tp4 => TpFilter::All,
        }
    }

    pub fn matches(&self, model: &llmfit_core::models::LlmModel) -> bool {
        match self {
            TpFilter::All => true,
            TpFilter::Tp2 => model.supports_tp(2),
            TpFilter::Tp3 => model.supports_tp(3),
            TpFilter::Tp4 => model.supports_tp(4),
        }
    }

    fn from_label(label: &str) -> Self {
        match label {
            "TP=2" => TpFilter::Tp2,
            "TP=3" => TpFilter::Tp3,
            "TP=4" => TpFilter::Tp4,
            _ => TpFilter::All,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextFilter {
    All,
    AtLeast32k,
    AtLeast40k,
    AtLeast128k,
    AtLeast131k,
    AtLeast262k,
    AtLeast512k,
    AtLeast1m,
}

impl ContextFilter {
    pub fn label(&self) -> &str {
        match self {
            ContextFilter::All => "All",
            ContextFilter::AtLeast32k => ">=32k",
            ContextFilter::AtLeast40k => ">=40k",
            ContextFilter::AtLeast128k => ">=128k",
            ContextFilter::AtLeast131k => ">=131k",
            ContextFilter::AtLeast262k => ">=262k",
            ContextFilter::AtLeast512k => ">=512k",
            ContextFilter::AtLeast1m => ">=1M",
        }
    }

    pub fn next(self) -> Self {
        match self {
            ContextFilter::All => ContextFilter::AtLeast32k,
            ContextFilter::AtLeast32k => ContextFilter::AtLeast40k,
            ContextFilter::AtLeast40k => ContextFilter::AtLeast128k,
            ContextFilter::AtLeast128k => ContextFilter::AtLeast131k,
            ContextFilter::AtLeast131k => ContextFilter::AtLeast262k,
            ContextFilter::AtLeast262k => ContextFilter::AtLeast512k,
            ContextFilter::AtLeast512k => ContextFilter::AtLeast1m,
            ContextFilter::AtLeast1m => ContextFilter::All,
        }
    }

    pub fn min_context(self) -> Option<u32> {
        match self {
            ContextFilter::All => None,
            ContextFilter::AtLeast32k => Some(32_768),
            ContextFilter::AtLeast40k => Some(40_960),
            ContextFilter::AtLeast128k | ContextFilter::AtLeast131k => Some(131_072),
            ContextFilter::AtLeast262k => Some(262_144),
            ContextFilter::AtLeast512k => Some(524_288),
            ContextFilter::AtLeast1m => Some(1_048_576),
        }
    }

    fn from_label(label: &str) -> Self {
        match label {
            ">=32k" => ContextFilter::AtLeast32k,
            ">=40k" => ContextFilter::AtLeast40k,
            ">=128k" => ContextFilter::AtLeast128k,
            ">=131k" => ContextFilter::AtLeast131k,
            ">=262k" => ContextFilter::AtLeast262k,
            ">=512k" => ContextFilter::AtLeast512k,
            ">=1M" => ContextFilter::AtLeast1m,
            _ => ContextFilter::All,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeFilter {
    Any,
    LlamaCpp,
    Mlx,
    Vllm,
    LmStudio,
}

impl RuntimeFilter {
    pub fn label(&self) -> &str {
        match self {
            RuntimeFilter::Any => "Any",
            RuntimeFilter::LlamaCpp => "llama.cpp",
            RuntimeFilter::Mlx => "MLX",
            RuntimeFilter::Vllm => "vLLM",
            RuntimeFilter::LmStudio => "LM Studio",
        }
    }

    pub fn next(self) -> Self {
        match self {
            RuntimeFilter::Any => RuntimeFilter::LlamaCpp,
            RuntimeFilter::LlamaCpp => RuntimeFilter::Mlx,
            RuntimeFilter::Mlx => RuntimeFilter::Vllm,
            RuntimeFilter::Vllm => RuntimeFilter::LmStudio,
            RuntimeFilter::LmStudio => RuntimeFilter::Any,
        }
    }

    fn from_label(label: &str) -> Self {
        match label {
            "llama.cpp" => RuntimeFilter::LlamaCpp,
            "MLX" => RuntimeFilter::Mlx,
            "vLLM" => RuntimeFilter::Vllm,
            "LM Studio" => RuntimeFilter::LmStudio,
            _ => RuntimeFilter::Any,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct FilterState {
    fit_filter: String,
    runtime_filter: String,
    availability_filter: String,
    tp_filter: String,
    context_filter: String,
    installed_first: bool,
    sort_column: String,
    sort_ascending: bool,
    selected_providers: Option<Vec<String>>,
    selected_use_cases: Option<Vec<String>>,
    selected_capabilities: Option<Vec<String>>,
    selected_quants: Option<Vec<String>>,
    selected_run_modes: Option<Vec<String>>,
    selected_params_buckets: Option<Vec<String>>,
}

impl Default for FilterState {
    fn default() -> Self {
        Self {
            fit_filter: FitFilter::All.label().to_string(),
            runtime_filter: RuntimeFilter::Any.label().to_string(),
            availability_filter: AvailabilityFilter::All.label().to_string(),
            tp_filter: TpFilter::All.label().to_string(),
            context_filter: ContextFilter::All.label().to_string(),
            installed_first: false,
            sort_column: SortColumn::Score.label().to_string(),
            sort_ascending: false,
            selected_providers: None,
            selected_use_cases: None,
            selected_capabilities: None,
            selected_quants: None,
            selected_run_modes: None,
            selected_params_buckets: None,
        }
    }
}

impl FilterState {
    fn config_path() -> Option<PathBuf> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()?;
        Some(
            PathBuf::from(home)
                .join(".config")
                .join("llmfit")
                .join("filters.json"),
        )
    }

    fn load() -> Option<Self> {
        Self::config_path().and_then(|path| Self::load_from_path(&path))
    }

    fn load_from_path(path: &Path) -> Option<Self> {
        let content = fs::read_to_string(path).ok()?;
        serde_json::from_str(&content).ok()
    }

    fn save(&self) {
        if let Some(path) = Self::config_path() {
            self.save_to_path(&path);
        }
    }

    fn save_to_path(&self, path: &Path) {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(self) {
            let _ = fs::write(path, json);
        }
    }

    fn from_app(app: &App) -> Self {
        Self {
            fit_filter: app.fit_filter.label().to_string(),
            runtime_filter: app.runtime_filter.label().to_string(),
            availability_filter: app.availability_filter.label().to_string(),
            tp_filter: app.tp_filter.label().to_string(),
            context_filter: app.context_filter.label().to_string(),
            installed_first: app.installed_first,
            sort_column: app.sort_column.label().to_string(),
            sort_ascending: app.sort_ascending,
            selected_providers: Some(App::selected_string_items(
                &app.providers,
                &app.selected_providers,
            )),
            selected_use_cases: Some(App::selected_labeled_items(
                &app.use_cases,
                &app.selected_use_cases,
                |use_case| use_case.label(),
            )),
            selected_capabilities: Some(App::selected_labeled_items(
                &app.capabilities,
                &app.selected_capabilities,
                |capability| capability.label(),
            )),
            selected_quants: Some(App::selected_string_items(&app.quants, &app.selected_quants)),
            selected_run_modes: Some(App::selected_string_items(
                &app.run_modes,
                &app.selected_run_modes,
            )),
            selected_params_buckets: Some(App::selected_string_items(
                &app.params_buckets,
                &app.selected_params_buckets,
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadProvider {
    Ollama,
    Mlx,
    LlamaCpp,
    DockerModelRunner,
    LmStudio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadCapability {
    Unknown,
    /// Bitfield: OLLAMA=1, LLAMACPP=2, DOCKER=4
    Known(u8),
}

pub const DL_OLLAMA: u8 = 0b0001;
pub const DL_LLAMACPP: u8 = 0b0010;
pub const DL_DOCKER: u8 = 0b0100;
pub const DL_LMSTUDIO: u8 = 0b1000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActivePullProvider {
    Ollama,
    Mlx,
    LlamaCpp,
    DockerModelRunner,
    LmStudio,
}

impl ActivePullProvider {
    fn label(self) -> &'static str {
        match self {
            ActivePullProvider::Ollama => "Ollama",
            ActivePullProvider::Mlx => "MLX",
            ActivePullProvider::LlamaCpp => "llama.cpp",
            ActivePullProvider::DockerModelRunner => "Docker",
            ActivePullProvider::LmStudio => "LM Studio",
        }
    }
}

enum CatalogRefreshEvent {
    Progress(String),
    Done(Result<(usize, usize), String>),
}

pub struct App {
    pub should_quit: bool,
    pub input_mode: InputMode,
    pub search_query: String,
    pub cursor_position: usize,

    // Data
    pub specs: SystemSpecs,
    source_models: Vec<LlmModel>,
    base_context_limit: Option<u32>,
    pub all_fits: Vec<ModelFit>,
    pub filtered_fits: Vec<usize>, // indices into all_fits
    pub providers: Vec<String>,
    pub selected_providers: Vec<bool>,
    pub use_cases: Vec<UseCase>,
    pub selected_use_cases: Vec<bool>,
    pub capabilities: Vec<Capability>,
    pub selected_capabilities: Vec<bool>,

    // Filters
    pub fit_filter: FitFilter,
    pub runtime_filter: RuntimeFilter,
    pub availability_filter: AvailabilityFilter,
    pub tp_filter: TpFilter,
    pub context_filter: ContextFilter,
    pub installed_first: bool,
    pub sort_column: SortColumn,
    pub sort_ascending: bool,

    // Table state
    pub selected_row: usize,

    // Detail view
    pub show_detail: bool,
    pub show_compare: bool,
    pub compare_mark_model: Option<String>,
    pub show_multi_compare: bool,
    pub compare_models: Vec<usize>, // indices into all_fits
    pub compare_scroll: usize,      // horizontal scroll for multi-compare
    pub show_plan: bool,
    plan_model_idx: Option<usize>,
    pub plan_field: PlanField,
    pub plan_context_input: String,
    pub plan_quant_input: String,
    pub plan_target_tps_input: String,
    pub plan_cursor_position: usize,
    pub plan_estimate: Option<PlanEstimate>,
    pub plan_error: Option<String>,

    // Provider popup
    pub provider_cursor: usize,
    pub use_case_cursor: usize,
    pub capability_cursor: usize,
    pub download_provider_cursor: usize,
    pub download_provider_options: Vec<DownloadProvider>,
    pub download_provider_model: Option<String>,

    // Provider state
    pub ollama_available: bool,
    pub ollama_binary_available: bool,
    pub ollama_installed: HashSet<String>,
    pub ollama_installed_count: usize,
    ollama: OllamaProvider,
    pub mlx_available: bool,
    pub mlx_installed: HashSet<String>,
    mlx: MlxProvider,
    pub llamacpp_available: bool,
    pub llamacpp_installed: HashSet<String>,
    pub llamacpp_installed_count: usize,
    pub llamacpp_detection_hint: String,
    llamacpp: LlamaCppProvider,
    pub docker_mr_available: bool,
    pub docker_mr_installed: HashSet<String>,
    pub docker_mr_installed_count: usize,
    docker_mr: DockerModelRunnerProvider,
    pub lmstudio_available: bool,
    pub lmstudio_installed: HashSet<String>,
    pub lmstudio_installed_count: usize,
    lmstudio: LmStudioProvider,
    pub vllm_available: bool,
    pub vllm_installed: HashSet<String>,
    pub vllm_installed_count: usize,
    vllm: VllmProvider,

    // Download state
    pub pull_active: Option<PullHandle>,
    pub pull_status: Option<String>,
    pub pull_percent: Option<f64>,
    pub pull_model_name: Option<String>,
    pull_provider: Option<ActivePullProvider>,
    pub download_capabilities: HashMap<String, DownloadCapability>,
    download_capability_inflight: HashSet<String>,
    download_capability_tx: mpsc::Sender<(String, DownloadCapability)>,
    download_capability_rx: mpsc::Receiver<(String, DownloadCapability)>,
    catalog_refresh_active: bool,
    catalog_refresh_tx: mpsc::Sender<CatalogRefreshEvent>,
    catalog_refresh_rx: mpsc::Receiver<CatalogRefreshEvent>,
    /// Animation frame counter, incremented every tick while pulling.
    pub tick_count: u64,
    /// When true, the next 'd' press will confirm and start the download.
    pub confirm_download: bool,

    // Visual mode
    pub visual_anchor: Option<usize>,

    // Select mode
    pub select_column: usize,

    // Quant filter (popup)
    pub quants: Vec<String>,
    pub selected_quants: Vec<bool>,
    pub quant_cursor: usize,

    // RunMode filter (popup)
    pub run_modes: Vec<String>,
    pub selected_run_modes: Vec<bool>,
    pub run_mode_cursor: usize,

    // Params bucket filter (popup)
    pub params_buckets: Vec<String>,
    pub selected_params_buckets: Vec<bool>,
    pub params_bucket_cursor: usize,

    // Theme
    pub theme: Theme,

    /// How many models we silently dropped because they can't run on this
    /// hardware — shown in the system bar so users aren't left wondering
    /// why the list looks shorter than expected.
    pub backend_hidden_count: usize,
}

impl App {
    fn selected_string_items(items: &[String], selected: &[bool]) -> Vec<String> {
        items
            .iter()
            .zip(selected.iter())
            .filter(|(_, is_selected)| **is_selected)
            .map(|(item, _)| item.clone())
            .collect()
    }

    fn selected_labeled_items<T>(
        items: &[T],
        selected: &[bool],
        label: impl Fn(&T) -> &'static str,
    ) -> Vec<String> {
        items
            .iter()
            .zip(selected.iter())
            .filter(|(_, is_selected)| **is_selected)
            .map(|(item, _)| label(item).to_string())
            .collect()
    }

    fn apply_saved_string_selection(items: &[String], saved: Option<&[String]>) -> Option<Vec<bool>> {
        saved.map(|selected_items| {
            items
                .iter()
                .map(|item| selected_items.iter().any(|selected| selected == item))
                .collect()
        })
    }

    fn apply_saved_labeled_selection<T>(
        items: &[T],
        saved: Option<&[String]>,
        label: impl Fn(&T) -> &'static str,
    ) -> Option<Vec<bool>> {
        saved.map(|selected_items| {
            items
                .iter()
                .map(|item| selected_items.iter().any(|selected| selected == label(item)))
                .collect()
        })
    }

    fn parse_sort_column(label: &str) -> SortColumn {
        match label {
            "tok/s" => SortColumn::Tps,
            "Params" => SortColumn::Params,
            "Mem%" => SortColumn::MemPct,
            "Ctx" => SortColumn::Ctx,
            "Date" => SortColumn::ReleaseDate,
            "Use" => SortColumn::UseCase,
            _ => SortColumn::Score,
        }
    }

    fn save_filter_state(&self) {
        FilterState::from_app(self).save();
    }

    fn reset_named_selection(selection: &mut [bool]) {
        for selected in selection {
            *selected = true;
        }
    }

    fn apply_filter_state(&mut self, state: &FilterState) {
        self.fit_filter = FitFilter::from_label(&state.fit_filter);
        self.runtime_filter = RuntimeFilter::from_label(&state.runtime_filter);
        self.availability_filter = AvailabilityFilter::from_label(&state.availability_filter);
        self.tp_filter = TpFilter::from_label(&state.tp_filter);
        self.context_filter = ContextFilter::from_label(&state.context_filter);
        self.installed_first = state.installed_first;
        self.sort_column = Self::parse_sort_column(&state.sort_column);
        self.sort_ascending = state.sort_ascending;

        if let Some(selected) = Self::apply_saved_string_selection(
            &self.providers,
            state.selected_providers.as_deref(),
        ) {
            self.selected_providers = selected;
        }
        if let Some(selected) = Self::apply_saved_labeled_selection(
            &self.use_cases,
            state.selected_use_cases.as_deref(),
            |use_case| use_case.label(),
        ) {
            self.selected_use_cases = selected;
        }
        if let Some(selected) = Self::apply_saved_labeled_selection(
            &self.capabilities,
            state.selected_capabilities.as_deref(),
            |capability| capability.label(),
        ) {
            self.selected_capabilities = selected;
        }
        if let Some(selected) = Self::apply_saved_string_selection(
            &self.quants,
            state.selected_quants.as_deref(),
        ) {
            self.selected_quants = selected;
        }
        if let Some(selected) = Self::apply_saved_string_selection(
            &self.run_modes,
            state.selected_run_modes.as_deref(),
        ) {
            self.selected_run_modes = selected;
        }
        if let Some(selected) = Self::apply_saved_string_selection(
            &self.params_buckets,
            state.selected_params_buckets.as_deref(),
        ) {
            self.selected_params_buckets = selected;
        }
    }

    fn preserve_string_selection(
        previous_items: &[String],
        previous_selected: &[bool],
        next_items: &[String],
    ) -> Vec<bool> {
        next_items
            .iter()
            .map(|item| {
                previous_items
                    .iter()
                    .position(|prev| prev == item)
                    .and_then(|idx| previous_selected.get(idx).copied())
                    .unwrap_or(true)
            })
            .collect()
    }

    fn preserve_copy_selection<T: Copy + PartialEq>(
        previous_items: &[T],
        previous_selected: &[bool],
        next_items: &[T],
    ) -> Vec<bool> {
        next_items
            .iter()
            .map(|item| {
                previous_items
                    .iter()
                    .position(|prev| prev == item)
                    .and_then(|idx| previous_selected.get(idx).copied())
                    .unwrap_or(true)
            })
            .collect()
    }

    fn build_fits(
        models: &[LlmModel],
        specs: &SystemSpecs,
        context_limit: Option<u32>,
        ollama_installed: &HashSet<String>,
        mlx_installed: &HashSet<String>,
        llamacpp_installed: &HashSet<String>,
        docker_mr_installed: &HashSet<String>,
        lmstudio_installed: &HashSet<String>,
        vllm_installed: &HashSet<String>,
    ) -> Vec<ModelFit> {
        models
            .iter()
            .map(|m| {
                let mut fit = ModelFit::analyze_with_context_limit(m, specs, context_limit);
                fit.installed = providers::is_model_installed(&m.name, ollama_installed)
                    || providers::is_model_installed_mlx(&m.name, mlx_installed)
                    || providers::is_model_installed_llamacpp(&m.name, llamacpp_installed)
                    || providers::is_model_installed_docker_mr(&m.name, docker_mr_installed)
                    || providers::is_model_installed_lmstudio(&m.name, lmstudio_installed)
                    || providers::is_model_installed_vllm(&m.name, vllm_installed);
                fit
            })
            .collect()
    }

    fn sync_filter_options_from_all_fits(&mut self) {
        let mut model_providers: Vec<String> = self
            .all_fits
            .iter()
            .map(|f| f.model.provider.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        model_providers.sort();
        self.selected_providers = Self::preserve_string_selection(
            &self.providers,
            &self.selected_providers,
            &model_providers,
        );
        self.providers = model_providers;

        let model_use_cases = [
            UseCase::General,
            UseCase::Coding,
            UseCase::Reasoning,
            UseCase::Chat,
            UseCase::Agentic,
            UseCase::Multimodal,
            UseCase::Embedding,
        ]
        .into_iter()
        .filter(|uc| self.all_fits.iter().any(|f| f.use_case == *uc))
        .collect::<Vec<_>>();
        self.selected_use_cases = Self::preserve_copy_selection(
            &self.use_cases,
            &self.selected_use_cases,
            &model_use_cases,
        );
        self.use_cases = model_use_cases;

        let model_capabilities = Capability::all().to_vec();
        self.selected_capabilities = Self::preserve_copy_selection(
            &self.capabilities,
            &self.selected_capabilities,
            &model_capabilities,
        );
        self.capabilities = model_capabilities;

        let mut model_quants: Vec<String> = self
            .all_fits
            .iter()
            .map(|f| f.best_quant.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        model_quants.sort();
        self.selected_quants =
            Self::preserve_string_selection(&self.quants, &self.selected_quants, &model_quants);
        self.quants = model_quants;
    }

    fn runtime_filter_matches_fit(&self, fit: &ModelFit) -> bool {
        match self.runtime_filter {
            RuntimeFilter::Any => true,
            RuntimeFilter::LlamaCpp => {
                fit.runtime == InferenceRuntime::LlamaCpp
                    && (fit.model.format == llmfit_core::models::ModelFormat::Gguf
                        || !fit.model.gguf_sources.is_empty()
                        || providers::is_model_installed_llamacpp(
                            &fit.model.name,
                            &self.llamacpp_installed,
                        ))
            }
            RuntimeFilter::Mlx => fit.runtime == InferenceRuntime::Mlx,
            RuntimeFilter::Vllm => {
                fit.runtime == InferenceRuntime::Vllm
                    || providers::is_model_installed_vllm(&fit.model.name, &self.vllm_installed)
            }
            RuntimeFilter::LmStudio => {
                providers::is_model_installed_lmstudio(
                    &fit.model.name,
                    &self.lmstudio_installed,
                )
            }
        }
    }

    pub fn with_specs_and_context(specs: SystemSpecs, context_limit: Option<u32>) -> Self {
        let db = ModelDatabase::new();

        // Detect Ollama
        let ollama = OllamaProvider::new();
        let (ollama_available, ollama_installed, ollama_installed_count) =
            ollama.detect_with_installed();
        let ollama_binary_available = command_exists("ollama");

        // Detect MLX
        let mlx = MlxProvider::new();
        let (mlx_available, mlx_installed) = mlx.detect_with_installed();

        // Detect llama.cpp
        let llamacpp = LlamaCppProvider::new();
        let llamacpp_available = llamacpp.is_available();
        let llamacpp_detection_hint = llamacpp.detection_hint().to_string();
        let (llamacpp_installed, llamacpp_installed_count) = llamacpp.installed_models_counted();

        // Detect Docker Model Runner
        let docker_mr = DockerModelRunnerProvider::new();
        let (docker_mr_available, docker_mr_installed, docker_mr_installed_count) =
            docker_mr.detect_with_installed();

        // Detect LM Studio
        let lmstudio = LmStudioProvider::new();
        let (lmstudio_available, lmstudio_installed, lmstudio_installed_count) =
            lmstudio.detect_with_installed();

        let vllm = VllmProvider::new();
        let (vllm_available, vllm_installed, vllm_installed_count) = vllm.detect_with_installed();

        // Track how many we're skipping so the UI can surface it.
        let backend_hidden_count = db
            .get_all_models()
            .iter()
            .filter(|m| !backend_compatible(m, &specs))
            .count();

        // Only analyze models that can actually run on this hardware.
        let source_models: Vec<LlmModel> = db
            .get_all_models()
            .iter()
            .filter(|m| backend_compatible(m, &specs))
            .cloned()
            .collect();

        let mut all_fits = Self::build_fits(
            &source_models,
            &specs,
            context_limit,
            &ollama_installed,
            &mlx_installed,
            &llamacpp_installed,
            &docker_mr_installed,
            &lmstudio_installed,
            &vllm_installed,
        );

        // Sort by fit level then RAM usage
        all_fits = llmfit_core::fit::rank_models_by_fit(all_fits);

        // Extract unique providers
        let mut model_providers: Vec<String> = all_fits
            .iter()
            .map(|f| f.model.provider.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        model_providers.sort();

        let selected_providers = vec![true; model_providers.len()];
        let model_use_cases = [
            UseCase::General,
            UseCase::Coding,
            UseCase::Reasoning,
            UseCase::Chat,
            UseCase::Agentic,
            UseCase::Multimodal,
            UseCase::Embedding,
        ]
        .into_iter()
        .filter(|uc| all_fits.iter().any(|f| f.use_case == *uc))
        .collect::<Vec<_>>();
        let selected_use_cases = vec![true; model_use_cases.len()];

        let model_capabilities = Capability::all().to_vec();
        let selected_capabilities = vec![true; model_capabilities.len()];

        // Extract unique quantizations
        let mut model_quants: Vec<String> = all_fits
            .iter()
            .map(|f| f.best_quant.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        model_quants.sort();
        let selected_quants = vec![true; model_quants.len()];

        // Run modes
        let model_run_modes = vec![
            "GPU".to_string(),
            "MoE".to_string(),
            "CPU+GPU".to_string(),
            "CPU".to_string(),
        ];
        let selected_run_modes = vec![true; model_run_modes.len()];

        // Params buckets
        let params_buckets = vec![
            "<3B".to_string(),
            "3-7B".to_string(),
            "7-14B".to_string(),
            "14-30B".to_string(),
            "30-70B".to_string(),
            "70B+".to_string(),
        ];
        let selected_params_buckets = vec![true; params_buckets.len()];

        let filtered_count = all_fits.len();

        let (download_capability_tx, download_capability_rx) = mpsc::channel();
        let (catalog_refresh_tx, catalog_refresh_rx) = mpsc::channel();

        let saved_filter_state = FilterState::load();

        let mut app = App {
            should_quit: false,
            input_mode: InputMode::Normal,
            search_query: String::new(),
            cursor_position: 0,
            specs,
            source_models,
            base_context_limit: context_limit,
            all_fits,
            filtered_fits: (0..filtered_count).collect(),
            providers: model_providers,
            selected_providers,
            use_cases: model_use_cases,
            selected_use_cases,
            capabilities: model_capabilities,
            selected_capabilities,
            fit_filter: FitFilter::All,
            runtime_filter: RuntimeFilter::Any,
            availability_filter: AvailabilityFilter::All,
            tp_filter: TpFilter::All,
            context_filter: ContextFilter::All,
            installed_first: false,
            sort_column: SortColumn::Score,
            sort_ascending: false,
            selected_row: 0,
            show_detail: false,
            show_compare: false,
            compare_mark_model: None,
            show_multi_compare: false,
            compare_models: Vec::new(),
            compare_scroll: 0,
            show_plan: false,
            plan_model_idx: None,
            plan_field: PlanField::Context,
            plan_context_input: String::new(),
            plan_quant_input: String::new(),
            plan_target_tps_input: String::new(),
            plan_cursor_position: 0,
            plan_estimate: None,
            plan_error: None,
            provider_cursor: 0,
            use_case_cursor: 0,
            capability_cursor: 0,
            download_provider_cursor: 0,
            download_provider_options: Vec::new(),
            download_provider_model: None,
            ollama_available,
            ollama_binary_available,
            ollama_installed,
            ollama_installed_count,
            ollama,
            mlx_available,
            mlx_installed,
            mlx,
            llamacpp_available,
            llamacpp_installed,
            llamacpp_installed_count,
            llamacpp_detection_hint,
            llamacpp,
            docker_mr_available,
            docker_mr_installed,
            docker_mr_installed_count,
            docker_mr,
            lmstudio_available,
            lmstudio_installed,
            lmstudio_installed_count,
            lmstudio,
            vllm_available,
            vllm_installed,
            vllm_installed_count,
            vllm,
            pull_active: None,
            pull_status: None,
            pull_percent: None,
            pull_model_name: None,
            pull_provider: None,
            download_capabilities: HashMap::new(),
            download_capability_inflight: HashSet::new(),
            download_capability_tx,
            download_capability_rx,
            catalog_refresh_active: false,
            catalog_refresh_tx,
            catalog_refresh_rx,
            tick_count: 0,
            confirm_download: false,
            visual_anchor: None,
            select_column: 2, // start on Model column
            quants: model_quants,
            selected_quants,
            quant_cursor: 0,
            run_modes: model_run_modes,
            selected_run_modes,
            run_mode_cursor: 0,
            params_buckets,
            selected_params_buckets,
            params_bucket_cursor: 0,
            theme: Theme::load(),
            backend_hidden_count,
        };

        if let Some(filter_state) = saved_filter_state {
            app.apply_filter_state(&filter_state);
            app.rebuild_fits();
            app.apply_filter_state(&filter_state);
            app.apply_filters();
        } else {
            app.apply_filters();
        }
        app.enqueue_capability_probes_for_visible(24);
        app
    }

    pub fn apply_filters(&mut self) {
        let query = self.search_query.to_lowercase();
        // Split query into space-separated terms for fuzzy matching
        let terms: Vec<&str> = query.split_whitespace().collect();

        self.filtered_fits = self
            .all_fits
            .iter()
            .enumerate()
            .filter(|(_, fit)| {
                // Search filter: all terms must match (fuzzy/AND logic)
                let matches_search = if terms.is_empty() {
                    true
                } else {
                    let effective_capabilities = fit.model.effective_capabilities();
                    let effective_context_length = fit.model.effective_context_length();
                    let caps_text = effective_capabilities
                        .iter()
                        .map(|c| c.label().to_lowercase())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let context_text = format!(
                        "{} {}",
                        effective_context_length,
                        llmfit_core::models::format_context_length(effective_context_length)
                            .to_lowercase()
                    );
                    // Combine all searchable fields into one string
                    let searchable = format!(
                        "{} {} {} {} {} {} {}",
                        fit.model.name.to_lowercase(),
                        fit.model.provider.to_lowercase(),
                        fit.model.parameter_count.to_lowercase(),
                        fit.model.effective_use_case().to_lowercase(),
                        fit.use_case.label().to_lowercase(),
                        caps_text,
                        context_text,
                    );
                    // All terms must be present (AND logic)
                    terms.iter().all(|term| {
                        searchable.contains(term)
                            || llmfit_core::models::context_matches_search_term(
                                term,
                                effective_context_length,
                            )
                    })
                };

                // Provider filter
                let provider_idx = self.providers.iter().position(|p| p == &fit.model.provider);
                let matches_provider = provider_idx
                    .map(|idx| self.selected_providers[idx])
                    .unwrap_or(true);
                let use_case_idx = self.use_cases.iter().position(|uc| *uc == fit.use_case);
                let matches_use_case = use_case_idx
                    .map(|idx| self.selected_use_cases[idx])
                    .unwrap_or(true);

                // Hide MLX-only models on non-Apple Silicon systems
                let is_apple_silicon = self.specs.backend
                    == llmfit_core::hardware::GpuBackend::Metal
                    && self.specs.unified_memory;
                if fit.model.is_mlx_only() && !is_apple_silicon {
                    return false;
                }

                // Fit filter
                let matches_fit = match self.fit_filter {
                    FitFilter::All => true,
                    FitFilter::Perfect => fit.fit_level == FitLevel::Perfect,
                    FitFilter::Good => fit.fit_level == FitLevel::Good,
                    FitFilter::Marginal => fit.fit_level == FitLevel::Marginal,
                    FitFilter::TooTight => fit.fit_level == FitLevel::TooTight,
                    FitFilter::Runnable => fit.fit_level != FitLevel::TooTight,
                };

                let matches_runtime = self.runtime_filter_matches_fit(fit);

                // Availability filter
                let matches_availability = match self.availability_filter {
                    AvailabilityFilter::All => true,
                    AvailabilityFilter::HasGguf => !fit.model.gguf_sources.is_empty(),
                    AvailabilityFilter::Installed => fit.installed,
                };

                // Capability filter
                let matches_capability = {
                    let all_selected = self.selected_capabilities.iter().all(|&s| s);
                    if all_selected {
                        true
                    } else {
                        self.capabilities
                            .iter()
                            .zip(self.selected_capabilities.iter())
                            .filter(|(_, sel)| **sel)
                            .any(|(cap, _)| fit.model.effective_capabilities().contains(cap))
                    }
                };

                // Quant filter
                let matches_quant = {
                    let all_selected = self.selected_quants.iter().all(|&s| s);
                    if all_selected {
                        true
                    } else {
                        self.quants
                            .iter()
                            .zip(self.selected_quants.iter())
                            .any(|(q, &sel)| sel && *q == fit.best_quant)
                    }
                };

                // RunMode filter
                let matches_run_mode = {
                    let all_selected = self.selected_run_modes.iter().all(|&s| s);
                    if all_selected {
                        true
                    } else {
                        let mode_text = fit.run_mode_text();
                        self.run_modes
                            .iter()
                            .zip(self.selected_run_modes.iter())
                            .any(|(m, &sel)| sel && *m == mode_text)
                    }
                };

                // Params bucket filter
                let matches_params_bucket = {
                    let all_selected = self.selected_params_buckets.iter().all(|&s| s);
                    if all_selected {
                        true
                    } else {
                        let params = fit.model.params_b();
                        let bucket_idx = if params < 3.0 {
                            0
                        } else if params < 7.0 {
                            1
                        } else if params < 14.0 {
                            2
                        } else if params < 30.0 {
                            3
                        } else if params < 70.0 {
                            4
                        } else {
                            5
                        };
                        self.selected_params_buckets
                            .get(bucket_idx)
                            .copied()
                            .unwrap_or(true)
                    }
                };

                let matches_tp = self.tp_filter.matches(&fit.model);
                let matches_context = self
                    .context_filter
                    .min_context()
                    .map(|min_context| fit.model.effective_context_length() >= min_context)
                    .unwrap_or(true);

                matches_search
                    && matches_provider
                    && matches_use_case
                    && matches_fit
                    && matches_runtime
                    && matches_availability
                    && matches_capability
                    && matches_quant
                    && matches_run_mode
                    && matches_params_bucket
                    && matches_tp
                    && matches_context
            })
            .map(|(i, _)| i)
            .collect();

        // Clamp selection
        if self.filtered_fits.is_empty() {
            self.selected_row = 0;
        } else if self.selected_row >= self.filtered_fits.len() {
            self.selected_row = self.filtered_fits.len() - 1;
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn selected_fit(&self) -> Option<&ModelFit> {
        self.filtered_fits
            .get(self.selected_row)
            .map(|&idx| &self.all_fits[idx])
    }

    pub fn move_up(&mut self) {
        self.confirm_download = false;
        if self.selected_row > 0 {
            self.selected_row -= 1;
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn move_down(&mut self) {
        self.confirm_download = false;
        if !self.filtered_fits.is_empty() && self.selected_row < self.filtered_fits.len() - 1 {
            self.selected_row += 1;
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn page_up(&mut self) {
        self.confirm_download = false;
        self.selected_row = self.selected_row.saturating_sub(10);
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn page_down(&mut self) {
        self.confirm_download = false;
        if !self.filtered_fits.is_empty() {
            self.selected_row = (self.selected_row + 10).min(self.filtered_fits.len() - 1);
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn half_page_up(&mut self) {
        self.selected_row = self.selected_row.saturating_sub(5);
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn half_page_down(&mut self) {
        if !self.filtered_fits.is_empty() {
            self.selected_row = (self.selected_row + 5).min(self.filtered_fits.len() - 1);
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn home(&mut self) {
        self.selected_row = 0;
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn end(&mut self) {
        if !self.filtered_fits.is_empty() {
            self.selected_row = self.filtered_fits.len() - 1;
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn cycle_fit_filter(&mut self) {
        self.fit_filter = self.fit_filter.next();
        self.apply_filters();
        self.save_filter_state();
    }

    pub fn cycle_runtime_filter(&mut self) {
        self.runtime_filter = self.runtime_filter.next();
        self.apply_filters();
        self.save_filter_state();
    }

    pub fn cycle_availability_filter(&mut self) {
        self.availability_filter = self.availability_filter.next();
        self.apply_filters();
        self.save_filter_state();
    }

    pub fn cycle_tp_filter(&mut self) {
        self.tp_filter = self.tp_filter.next();
        self.apply_filters();
        self.save_filter_state();
    }

    pub fn cycle_context_filter(&mut self) {
        self.context_filter = self.context_filter.next();
        self.rebuild_fits();
        self.save_filter_state();
    }

    pub fn cycle_sort_column(&mut self) {
        self.sort_column = self.sort_column.next();
        self.sort_ascending = false;
        self.re_sort();
        self.save_filter_state();
    }

    pub fn cycle_theme(&mut self) {
        self.theme = self.theme.next();
        self.theme.save();
    }

    pub fn reset_filters(&mut self) {
        self.fit_filter = FitFilter::All;
        self.runtime_filter = RuntimeFilter::Any;
        self.availability_filter = AvailabilityFilter::All;
        self.tp_filter = TpFilter::All;
        self.context_filter = ContextFilter::All;
        self.installed_first = false;
        self.sort_column = SortColumn::Score;
        self.sort_ascending = false;
        Self::reset_named_selection(&mut self.selected_providers);
        Self::reset_named_selection(&mut self.selected_use_cases);
        Self::reset_named_selection(&mut self.selected_capabilities);
        Self::reset_named_selection(&mut self.selected_quants);
        Self::reset_named_selection(&mut self.selected_run_modes);
        Self::reset_named_selection(&mut self.selected_params_buckets);
        self.search_query.clear();
        self.cursor_position = 0;
        self.rebuild_fits();
        self.save_filter_state();
        self.pull_status = Some("Reset all filters".to_string());
    }

    pub fn enter_search(&mut self) {
        self.input_mode = InputMode::Search;
    }

    pub fn exit_search(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn search_input(&mut self, c: char) {
        self.search_query.insert(self.cursor_position, c);
        self.cursor_position += 1;
        self.apply_filters();
    }

    pub fn search_backspace(&mut self) {
        if self.cursor_position > 0 {
            self.cursor_position -= 1;
            self.search_query.remove(self.cursor_position);
            self.apply_filters();
        }
    }

    pub fn search_delete(&mut self) {
        if self.cursor_position < self.search_query.len() {
            self.search_query.remove(self.cursor_position);
            self.apply_filters();
        }
    }

    pub fn clear_search(&mut self) {
        self.search_query.clear();
        self.cursor_position = 0;
        self.apply_filters();
    }

    pub fn toggle_detail(&mut self) {
        self.show_plan = false;
        self.show_compare = false;
        self.show_detail = !self.show_detail;
    }

    pub fn mark_selected_for_compare(&mut self) {
        let Some(model_name) = self.selected_fit().map(|fit| fit.model.name.clone()) else {
            self.pull_status = Some("No selected model to mark".to_string());
            return;
        };
        self.compare_mark_model = Some(model_name.clone());
        self.pull_status = Some(format!("Marked '{}' for compare", model_name));
    }

    pub fn clear_compare_mark(&mut self) {
        self.compare_mark_model = None;
        self.show_compare = false;
        self.pull_status = Some("Cleared compare mark".to_string());
    }

    pub fn copy_selected_model_name(&mut self) {
        let Some(fit) = self.selected_fit() else {
            self.pull_status = Some("No model selected".to_string());
            return;
        };
        let name = fit.model.name.clone();
        match arboard::Clipboard::new() {
            Ok(mut clipboard) => match clipboard.set_text(&name) {
                Ok(()) => self.pull_status = Some(format!("Copied '{}' to clipboard", name)),
                Err(e) => self.pull_status = Some(format!("Clipboard error: {}", e)),
            },
            Err(e) => self.pull_status = Some(format!("Clipboard error: {}", e)),
        }
    }

    pub fn selected_compare_pair(&self) -> Option<(&ModelFit, &ModelFit)> {
        let selected = self.selected_fit()?;
        let mark_name = self.compare_mark_model.as_deref()?;
        let marked = self.all_fits.iter().find(|f| f.model.name == mark_name)?;
        if marked.model.name == selected.model.name {
            return None;
        }
        Some((marked, selected))
    }

    pub fn toggle_compare_view(&mut self) {
        if self.show_compare {
            self.show_compare = false;
            return;
        }
        if self.compare_mark_model.is_none() {
            self.pull_status = Some("No marked model. Press m to mark one first".to_string());
            return;
        }
        if self.selected_compare_pair().is_none() {
            self.pull_status =
                Some("Select a different model than the marked one to compare".to_string());
            return;
        }
        self.show_detail = false;
        self.show_plan = false;
        self.show_compare = true;
    }

    pub fn open_plan_mode(&mut self) {
        let Some(&fit_idx) = self.filtered_fits.get(self.selected_row) else {
            return;
        };
        let fit = &self.all_fits[fit_idx];

        self.show_detail = false;
        self.show_compare = false;
        self.show_plan = true;
        self.input_mode = InputMode::Plan;
        self.plan_model_idx = Some(fit_idx);
        self.plan_field = PlanField::Context;
        self.plan_context_input = fit.model.context_length.min(8192).to_string();
        self.plan_quant_input = fit.model.quantization.clone();
        self.plan_target_tps_input.clear();
        self.plan_cursor_position = self.plan_context_input.len();
        self.refresh_plan_estimate();
    }

    pub fn close_plan_mode(&mut self) {
        self.show_plan = false;
        self.plan_model_idx = None;
        self.plan_estimate = None;
        self.plan_error = None;
        self.input_mode = InputMode::Normal;
    }

    pub fn plan_next_field(&mut self) {
        self.plan_field = self.plan_field.next();
        self.plan_cursor_position = self.active_plan_input().len();
    }

    pub fn plan_prev_field(&mut self) {
        self.plan_field = self.plan_field.prev();
        self.plan_cursor_position = self.active_plan_input().len();
    }

    pub fn plan_cursor_left(&mut self) {
        if self.plan_cursor_position > 0 {
            self.plan_cursor_position -= 1;
        }
    }

    pub fn plan_cursor_right(&mut self) {
        let len = self.active_plan_input().len();
        if self.plan_cursor_position < len {
            self.plan_cursor_position += 1;
        }
    }

    pub fn plan_input(&mut self, c: char) {
        match self.plan_field {
            PlanField::Context => {
                if !c.is_ascii_digit() {
                    return;
                }
            }
            PlanField::Quant => {
                if !(c.is_ascii_alphanumeric() || c == '_' || c == '-') {
                    return;
                }
            }
            PlanField::TargetTps => {
                if !(c.is_ascii_digit() || c == '.') {
                    return;
                }
                if c == '.' && self.plan_target_tps_input.contains('.') {
                    return;
                }
            }
        }

        let cursor = self.plan_cursor_position;
        let input = self.active_plan_input_mut();
        if cursor <= input.len() {
            input.insert(cursor, c);
            self.plan_cursor_position = cursor + 1;
            self.refresh_plan_estimate();
        }
    }

    pub fn plan_backspace(&mut self) {
        if self.plan_cursor_position == 0 {
            return;
        }
        let cursor = self.plan_cursor_position;
        let input = self.active_plan_input_mut();
        if cursor <= input.len() {
            input.remove(cursor - 1);
            self.plan_cursor_position = cursor - 1;
            self.refresh_plan_estimate();
        }
    }

    pub fn plan_delete(&mut self) {
        let cursor = self.plan_cursor_position;
        let input = self.active_plan_input_mut();
        if cursor < input.len() {
            input.remove(cursor);
            self.refresh_plan_estimate();
        }
    }

    pub fn plan_clear_field(&mut self) {
        self.active_plan_input_mut().clear();
        self.plan_cursor_position = 0;
        self.refresh_plan_estimate();
    }

    pub fn refresh_plan_estimate(&mut self) {
        let Some(model_idx) = self.plan_model_idx else {
            self.plan_estimate = None;
            self.plan_error = Some("No model selected for plan".to_string());
            return;
        };
        let Some(fit) = self.all_fits.get(model_idx) else {
            self.plan_estimate = None;
            self.plan_error = Some("Selected model is no longer available".to_string());
            return;
        };

        let context = match self.plan_context_input.trim().parse::<u32>() {
            Ok(v) if v > 0 => v,
            _ => {
                self.plan_estimate = None;
                self.plan_error = Some("Context must be a positive integer".to_string());
                return;
            }
        };

        let quant = if self.plan_quant_input.trim().is_empty() {
            None
        } else {
            Some(self.plan_quant_input.trim().to_string())
        };

        let target_tps = if self.plan_target_tps_input.trim().is_empty() {
            None
        } else {
            match self.plan_target_tps_input.trim().parse::<f64>() {
                Ok(v) if v > 0.0 => Some(v),
                _ => {
                    self.plan_estimate = None;
                    self.plan_error = Some("Target TPS must be a positive number".to_string());
                    return;
                }
            }
        };

        let request = PlanRequest {
            context,
            quant,
            target_tps,
        };

        match estimate_model_plan(&fit.model, &request, &self.specs) {
            Ok(plan) => {
                self.plan_estimate = Some(plan);
                self.plan_error = None;
            }
            Err(e) => {
                self.plan_estimate = None;
                self.plan_error = Some(e);
            }
        }
    }

    pub fn plan_model_name(&self) -> Option<&str> {
        self.plan_model_idx
            .and_then(|idx| self.all_fits.get(idx))
            .map(|fit| fit.model.name.as_str())
    }

    pub fn open_provider_popup(&mut self) {
        self.input_mode = InputMode::ProviderPopup;
        // Don't reset cursor -- keep it where it was last time
    }

    pub fn close_provider_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn open_use_case_popup(&mut self) {
        self.input_mode = InputMode::UseCasePopup;
        // Don't reset cursor -- keep it where it was last time
    }

    pub fn close_use_case_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn provider_popup_up(&mut self) {
        if self.provider_cursor > 0 {
            self.provider_cursor -= 1;
        }
    }

    pub fn provider_popup_down(&mut self) {
        if self.provider_cursor + 1 < self.providers.len() {
            self.provider_cursor += 1;
        }
    }

    pub fn provider_popup_toggle(&mut self) {
        if self.provider_cursor < self.selected_providers.len() {
            self.selected_providers[self.provider_cursor] =
                !self.selected_providers[self.provider_cursor];
            self.apply_filters();
            self.save_filter_state();
        }
    }

    pub fn provider_popup_select_all(&mut self) {
        let all_selected = self.selected_providers.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_providers {
            *s = new_val;
        }
        self.apply_filters();
        self.save_filter_state();
    }

    pub fn provider_popup_clear_all(&mut self) {
        for s in &mut self.selected_providers {
            *s = false;
        }
        self.apply_filters();
        self.save_filter_state();
    }

    pub fn use_case_popup_up(&mut self) {
        if self.use_case_cursor > 0 {
            self.use_case_cursor -= 1;
        }
    }

    pub fn use_case_popup_down(&mut self) {
        if self.use_case_cursor + 1 < self.use_cases.len() {
            self.use_case_cursor += 1;
        }
    }

    pub fn use_case_popup_toggle(&mut self) {
        if self.use_case_cursor < self.selected_use_cases.len() {
            self.selected_use_cases[self.use_case_cursor] =
                !self.selected_use_cases[self.use_case_cursor];
            self.apply_filters();
            self.save_filter_state();
        }
    }

    pub fn use_case_popup_select_all(&mut self) {
        let all_selected = self.selected_use_cases.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_use_cases {
            *s = new_val;
        }
        self.apply_filters();
        self.save_filter_state();
    }

    pub fn open_capability_popup(&mut self) {
        self.input_mode = InputMode::CapabilityPopup;
    }

    pub fn close_capability_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn capability_popup_up(&mut self) {
        if self.capability_cursor > 0 {
            self.capability_cursor -= 1;
        }
    }

    pub fn capability_popup_down(&mut self) {
        if self.capability_cursor + 1 < self.capabilities.len() {
            self.capability_cursor += 1;
        }
    }

    pub fn capability_popup_toggle(&mut self) {
        if self.capability_cursor < self.selected_capabilities.len() {
            self.selected_capabilities[self.capability_cursor] =
                !self.selected_capabilities[self.capability_cursor];
            self.apply_filters();
            self.save_filter_state();
        }
    }

    pub fn capability_popup_select_all(&mut self) {
        let all_selected = self.selected_capabilities.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_capabilities {
            *s = new_val;
        }
        self.apply_filters();
        self.save_filter_state();
    }

    // ── Visual mode ──────────────────────────────────────────────

    pub fn enter_visual_mode(&mut self) {
        self.visual_anchor = Some(self.selected_row);
        self.input_mode = InputMode::Visual;
    }

    pub fn exit_visual_mode(&mut self) {
        self.visual_anchor = None;
        self.input_mode = InputMode::Normal;
    }

    pub fn visual_range(&self) -> Option<std::ops::RangeInclusive<usize>> {
        let anchor = self.visual_anchor?;
        let lo = anchor.min(self.selected_row);
        let hi = anchor.max(self.selected_row);
        Some(lo..=hi)
    }

    pub fn visual_selection_count(&self) -> usize {
        self.visual_range()
            .map(|r| r.end() - r.start() + 1)
            .unwrap_or(0)
    }

    /// In visual mode, compare all selected models.
    pub fn visual_compare(&mut self) {
        let Some(range) = self.visual_range() else {
            return;
        };
        let lo = *range.start();
        let hi = *range.end();
        if lo == hi {
            self.pull_status = Some("Select at least 2 models to compare".to_string());
            return;
        }
        // Collect all filtered_fits indices in the visual range
        self.compare_models = (lo..=hi)
            .filter_map(|row| self.filtered_fits.get(row).copied())
            .collect();
        self.compare_scroll = 0;
        self.exit_visual_mode();
        self.show_detail = false;
        self.show_plan = false;
        self.show_compare = false;
        self.show_multi_compare = true;
    }

    pub fn close_multi_compare(&mut self) {
        self.show_multi_compare = false;
        self.compare_models.clear();
    }

    pub fn multi_compare_scroll_left(&mut self) {
        if self.compare_scroll > 0 {
            self.compare_scroll -= 1;
        }
    }

    pub fn multi_compare_scroll_right(&mut self) {
        if !self.compare_models.is_empty()
            && self.compare_scroll < self.compare_models.len().saturating_sub(1)
        {
            self.compare_scroll += 1;
        }
    }

    // ── Select mode ─────────────────────────────────────────────

    pub fn enter_select_mode(&mut self) {
        self.input_mode = InputMode::Select;
    }

    pub fn exit_select_mode(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn select_column_left(&mut self) {
        if self.select_column > 1 {
            self.select_column -= 1;
        }
    }

    pub fn select_column_right(&mut self) {
        if self.select_column < 13 {
            self.select_column += 1;
        }
    }

    /// Activate the filter for the currently focused column in Select mode.
    pub fn activate_select_column_filter(&mut self) {
        match self.select_column {
            1 => self.cycle_availability_filter(), // Inst
            2 => {
                self.input_mode = InputMode::Search;
            } // Model → search
            3 => {
                self.input_mode = InputMode::ProviderPopup;
            } // Provider
            4 => {
                self.input_mode = InputMode::ParamsBucketPopup;
            } // Params
            5 => self.set_or_toggle_sort(SortColumn::Score), // Score
            6 => self.set_or_toggle_sort(SortColumn::Tps), // tok/s
            7 => {
                self.input_mode = InputMode::QuantPopup;
            } // Quant
            8 => {
                self.input_mode = InputMode::RunModePopup;
            } // Mode
            9 => self.set_or_toggle_sort(SortColumn::MemPct), // Mem%
            10 => self.cycle_context_filter(),     // Ctx
            11 => self.set_or_toggle_sort(SortColumn::ReleaseDate), // Date
            12 => self.cycle_fit_filter(),         // Fit
            13 => {
                self.input_mode = InputMode::UseCasePopup;
            } // Use Case
            _ => {}
        }
    }

    /// Set sort column, or toggle ascending/descending if already on that column.
    fn set_or_toggle_sort(&mut self, col: SortColumn) {
        if self.sort_column == col {
            self.sort_ascending = !self.sort_ascending;
        } else {
            self.sort_column = col;
            self.sort_ascending = false;
        }
        self.re_sort();
        self.save_filter_state();
    }

    // ── Quant popup ─────────────────────────────────────────────

    pub fn close_quant_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn open_quant_popup(&mut self) {
        self.input_mode = InputMode::QuantPopup;
    }

    pub fn quant_popup_up(&mut self) {
        if self.quant_cursor > 0 {
            self.quant_cursor -= 1;
        }
    }

    pub fn quant_popup_down(&mut self) {
        if self.quant_cursor + 1 < self.quants.len() {
            self.quant_cursor += 1;
        }
    }

    pub fn quant_popup_toggle(&mut self) {
        if self.quant_cursor < self.selected_quants.len() {
            self.selected_quants[self.quant_cursor] = !self.selected_quants[self.quant_cursor];
            self.apply_filters();
            self.save_filter_state();
        }
    }

    pub fn quant_popup_select_all(&mut self) {
        let all_selected = self.selected_quants.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_quants {
            *s = new_val;
        }
        self.apply_filters();
        self.save_filter_state();
    }

    // ── RunMode popup ───────────────────────────────────────────

    pub fn close_run_mode_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn run_mode_popup_up(&mut self) {
        if self.run_mode_cursor > 0 {
            self.run_mode_cursor -= 1;
        }
    }

    pub fn run_mode_popup_down(&mut self) {
        if self.run_mode_cursor + 1 < self.run_modes.len() {
            self.run_mode_cursor += 1;
        }
    }

    pub fn run_mode_popup_toggle(&mut self) {
        if self.run_mode_cursor < self.selected_run_modes.len() {
            self.selected_run_modes[self.run_mode_cursor] =
                !self.selected_run_modes[self.run_mode_cursor];
            self.apply_filters();
            self.save_filter_state();
        }
    }

    pub fn run_mode_popup_select_all(&mut self) {
        let all_selected = self.selected_run_modes.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_run_modes {
            *s = new_val;
        }
        self.apply_filters();
        self.save_filter_state();
    }

    // ── Params bucket popup ─────────────────────────────────────

    pub fn close_params_bucket_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn params_bucket_popup_up(&mut self) {
        if self.params_bucket_cursor > 0 {
            self.params_bucket_cursor -= 1;
        }
    }

    pub fn params_bucket_popup_down(&mut self) {
        if self.params_bucket_cursor + 1 < self.params_buckets.len() {
            self.params_bucket_cursor += 1;
        }
    }

    pub fn params_bucket_popup_toggle(&mut self) {
        if self.params_bucket_cursor < self.selected_params_buckets.len() {
            self.selected_params_buckets[self.params_bucket_cursor] =
                !self.selected_params_buckets[self.params_bucket_cursor];
            self.apply_filters();
            self.save_filter_state();
        }
    }

    pub fn params_bucket_popup_select_all(&mut self) {
        let all_selected = self.selected_params_buckets.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_params_buckets {
            *s = new_val;
        }
        self.apply_filters();
        self.save_filter_state();
    }

    pub fn toggle_installed_first(&mut self) {
        self.installed_first = !self.installed_first;
        self.re_sort();
        self.save_filter_state();
    }

    /// Re-sort all_fits using current sort column and installed_first preference, then refilter.
    fn re_sort(&mut self) {
        let fits = std::mem::take(&mut self.all_fits);
        let mut sorted = llmfit_core::fit::rank_models_by_fit_opts_col(
            fits,
            self.installed_first,
            self.sort_column,
        );
        if self.sort_ascending {
            sorted.reverse();
        }
        self.all_fits = sorted;
        self.apply_filters();
    }

    fn effective_context_limit(&self) -> Option<u32> {
        match (self.base_context_limit, self.context_filter.min_context()) {
            (Some(base), Some(ctx)) => Some(base.min(ctx)),
            (Some(base), None) => Some(base),
            (None, Some(ctx)) => Some(ctx),
            (None, None) => None,
        }
    }

    fn rebuild_fits(&mut self) {
        let context_limit = self.effective_context_limit();
        let fits = Self::build_fits(
            &self.source_models,
            &self.specs,
            context_limit,
            &self.ollama_installed,
            &self.mlx_installed,
            &self.llamacpp_installed,
            &self.docker_mr_installed,
            &self.lmstudio_installed,
            &self.vllm_installed,
        );
        let mut sorted = llmfit_core::fit::rank_models_by_fit_opts_col(
            fits,
            self.installed_first,
            self.sort_column,
        );
        if self.sort_ascending {
            sorted.reverse();
        }
        self.all_fits = sorted;
        self.sync_filter_options_from_all_fits();
        self.apply_filters();
    }

    fn reload_model_catalog(&mut self) {
        let selected_name = self.selected_fit().map(|fit| fit.model.name.clone());
        let db = ModelDatabase::new();

        self.backend_hidden_count = db
            .get_all_models()
            .iter()
            .filter(|m| !backend_compatible(m, &self.specs))
            .count();

        self.source_models = db
            .get_all_models()
            .iter()
            .filter(|m| backend_compatible(m, &self.specs))
            .cloned()
            .collect();

        self.compare_models.clear();
        self.show_compare = false;
        self.show_multi_compare = false;
        self.compare_scroll = 0;
        self.plan_model_idx = None;
        self.show_plan = false;
        self.download_capabilities.clear();
        self.download_capability_inflight.clear();

        self.rebuild_fits();

        if let Some(name) = selected_name
            && let Some(idx) = self
                .filtered_fits
                .iter()
                .position(|fit_idx| self.all_fits[*fit_idx].model.name == name)
        {
            self.selected_row = idx;
        }
    }

    pub fn refresh_model_catalog(&mut self) {
        let specific_models = if self.show_detail {
            self.selected_fit()
                .map(|fit| vec![fit.model.name.clone()])
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        self.refresh_model_catalog_with_scope(specific_models);
    }

    fn refresh_model_catalog_with_scope(&mut self, specific_models: Vec<String>) {
        if self.catalog_refresh_active {
            self.pull_status = Some("Model refresh already running".to_string());
            return;
        }

        self.catalog_refresh_active = true;
        self.pull_status = Some(if specific_models.is_empty() {
            "Refreshing model catalog from HuggingFace...".to_string()
        } else {
            format!(
                "Refreshing {} from HuggingFace...",
                specific_models.join(", ")
            )
        });

        let tx = self.catalog_refresh_tx.clone();
        std::thread::spawn(move || {
            let opts = UpdateOptions {
                token: std::env::var("HF_TOKEN")
                    .ok()
                    .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok()),
                refresh_existing: true,
                trending_limit: if specific_models.is_empty() {
                    UpdateOptions::default().trending_limit
                } else {
                    0
                },
                downloads_limit: if specific_models.is_empty() {
                    UpdateOptions::default().downloads_limit
                } else {
                    0
                },
                specific_models,
                ..Default::default()
            };
            let result = update::update_model_cache(&opts, |msg| {
                let _ = tx.send(CatalogRefreshEvent::Progress(msg.to_string()));
            });
            let _ = tx.send(CatalogRefreshEvent::Done(result));
        });
    }

    fn tick_catalog_refresh(&mut self) {
        loop {
            match self.catalog_refresh_rx.try_recv() {
                Ok(CatalogRefreshEvent::Progress(status)) => {
                    self.pull_status = Some(status);
                }
                Ok(CatalogRefreshEvent::Done(Ok((changed, total)))) => {
                    self.catalog_refresh_active = false;
                    self.reload_model_catalog();
                    self.pull_status = Some(format!(
                        "Model catalog refreshed: {} updated/new, {} cached total",
                        changed, total
                    ));
                    self.enqueue_capability_probes_for_visible(24);
                    return;
                }
                Ok(CatalogRefreshEvent::Done(Err(err))) => {
                    self.catalog_refresh_active = false;
                    self.pull_status = Some(format!("Model refresh failed: {}", err));
                    return;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.catalog_refresh_active = false;
                    self.pull_status = Some("Model refresh ended".to_string());
                    return;
                }
            }
        }
    }

    /// Start pulling the currently selected model via the best available provider.
    pub fn start_download(&mut self) {
        let any_available = self.ollama_available
            || self.mlx_available
            || self.llamacpp_available
            || self.docker_mr_available
            || self.lmstudio_available
            || self.vllm_available;
        if !any_available {
            self.pull_status = Some(
                "No runtime available — install Ollama, llama.cpp, vLLM, Docker, or LM Studio"
                    .to_string(),
            );
            return;
        }
        if self.pull_active.is_some() {
            return; // already pulling
        }
        let Some(fit) = self.selected_fit() else {
            return;
        };
        if fit.installed {
            self.pull_status = Some("Already installed".to_string());
            return;
        }
        let model_name = fit.model.name.clone();
        let model_format = fit.model.format;
        let is_mlx_model = fit.model.is_mlx_model();
        let has_catalog_gguf = !fit.model.gguf_sources.is_empty();

        let download_options = self.available_download_providers(&model_name, has_catalog_gguf);
        if !download_options.is_empty() {
            self.open_download_provider_popup(model_name, download_options);
        } else {
            let any_runtime = self.ollama_available
                || self.ollama_binary_available
                || self.llamacpp_available
                || self.mlx_available
                || self.docker_mr_available
                || self.lmstudio_available
                || self.vllm_available;
            self.pull_status = Some(if any_runtime {
                Self::format_no_download_message(model_format, is_mlx_model, self.vllm_available)
            } else {
                "No runtime available — install Ollama, llama.cpp, vLLM, Docker, or LM Studio"
                    .to_string()
            });
        }
    }

    /// Build a user-friendly message explaining why no download is available,
    /// based on the model's weight format.
    fn format_no_download_message(
        format: llmfit_core::models::ModelFormat,
        is_mlx_model: bool,
        vllm_available: bool,
    ) -> String {
        use llmfit_core::models::ModelFormat;
        if is_mlx_model {
            "MLX model — requires Apple Silicon with MLX installed".to_string()
        } else {
            match format {
                ModelFormat::Awq => if vllm_available {
                    "AWQ model — vLLM is installed, but llmfit cannot download AWQ weights automatically yet"
                        .to_string()
                } else {
                    "AWQ model — requires vLLM on a CUDA/ROCm GPU; no GGUF conversion available"
                        .to_string()
                },
                ModelFormat::Gptq => if vllm_available {
                    "GPTQ model — vLLM is installed, but llmfit cannot download GPTQ weights automatically yet"
                        .to_string()
                } else {
                    "GPTQ model — requires vLLM on a CUDA/ROCm GPU; no GGUF conversion available"
                        .to_string()
                },
                _ => "No downloadable format found for this model".to_string(),
            }
        }
    }

    fn start_mlx_download(&mut self, model_name: String) {
        let tag = providers::mlx_pull_tag(&model_name);
        match self.mlx.start_pull(&tag) {
            Ok(handle) => {
                self.pull_model_name = Some(model_name);
                let repo_display = if tag.contains('/') {
                    tag
                } else {
                    format!("mlx-community/{}", tag)
                };
                self.pull_status = Some(format!("Pulling {}...", repo_display));
                self.pull_percent = None;
                self.pull_provider = Some(ActivePullProvider::Mlx);
                self.pull_active = Some(handle);
            }
            Err(e) => {
                self.pull_status = Some(format!("MLX pull failed: {}", e));
            }
        }
    }

    fn start_download_with_provider(&mut self, model_name: String, provider: DownloadProvider) {
        match provider {
            DownloadProvider::Ollama => self.start_ollama_download(model_name),
            DownloadProvider::Mlx => self.start_mlx_download(model_name),
            DownloadProvider::LlamaCpp => self.start_llamacpp_download_for_model(model_name),
            DownloadProvider::DockerModelRunner => self.start_docker_mr_download(model_name),
            DownloadProvider::LmStudio => self.start_lmstudio_download(model_name),
        }
    }

    fn start_ollama_download(&mut self, model_name: String) {
        let Some(tag) = providers::ollama_pull_tag(&model_name) else {
            self.pull_status = Some("Not available in Ollama registry".to_string());
            return;
        };
        match self.ollama.start_pull(&tag) {
            Ok(handle) => {
                self.pull_model_name = Some(model_name);
                self.pull_status = Some(format!("Pulling {}...", tag));
                self.pull_percent = Some(0.0);
                self.pull_provider = Some(ActivePullProvider::Ollama);
                self.pull_active = Some(handle);
            }
            Err(e) => {
                self.pull_status = Some(format!("Pull failed: {}", e));
            }
        }
    }

    /// Start downloading a GGUF model via the llama.cpp provider.
    fn start_llamacpp_download_for_model(&mut self, model_name: String) {
        // Check catalog gguf_sources first (instant), then fall back to HTTP probe
        let catalog_repo = self
            .all_fits
            .iter()
            .find(|f| f.model.name == model_name)
            .and_then(|f| f.model.gguf_sources.first())
            .map(|s| s.repo.clone());
        let Some(repo) = catalog_repo.or_else(|| providers::first_existing_gguf_repo(&model_name))
        else {
            self.pull_status = Some("No GGUF repo found in remote registry".to_string());
            return;
        };

        match self.llamacpp.start_pull(&repo) {
            Ok(handle) => {
                self.pull_model_name = Some(model_name);
                self.pull_status = Some(format!("Downloading GGUF from {}...", repo));
                self.pull_percent = Some(0.0);
                self.pull_provider = Some(ActivePullProvider::LlamaCpp);
                self.pull_active = Some(handle);
            }
            Err(e) => {
                self.pull_status = Some(format!("GGUF download failed: {}", e));
            }
        }
    }

    fn start_docker_mr_download(&mut self, model_name: String) {
        let Some(docker_tag) = providers::docker_mr_pull_tag(&model_name) else {
            self.pull_status = Some("Not available for Docker Model Runner".to_string());
            return;
        };
        match self.docker_mr.start_pull(&docker_tag) {
            Ok(handle) => {
                self.pull_model_name = Some(model_name);
                self.pull_status = Some(format!("Pulling {} via Docker...", docker_tag));
                self.pull_percent = None;
                self.pull_provider = Some(ActivePullProvider::DockerModelRunner);
                self.pull_active = Some(handle);
            }
            Err(e) => {
                self.pull_status = Some(format!("Docker pull failed: {}", e));
            }
        }
    }

    fn lmstudio_download_candidates(&self, model_name: &str) -> Vec<String> {
        self.all_fits
            .iter()
            .find(|fit| fit.model.name == model_name)
            .map(|fit| providers::lmstudio_download_candidates(model_name, &fit.model.gguf_sources))
            .unwrap_or_else(|| providers::lmstudio_download_candidates(model_name, &[]))
    }

    fn start_lmstudio_download(&mut self, model_name: String) {
        let candidates = self.lmstudio_download_candidates(&model_name);
        let Some(primary_tag) = candidates.first().cloned() else {
            self.pull_status = Some("Not available for LM Studio".to_string());
            return;
        };
        match self.lmstudio.start_pull_candidates(&candidates) {
            Ok(handle) => {
                self.pull_model_name = Some(model_name);
                self.pull_status = Some(format!("Downloading {} via LM Studio...", primary_tag));
                self.pull_percent = None;
                self.pull_provider = Some(ActivePullProvider::LmStudio);
                self.pull_active = Some(handle);
            }
            Err(e) => {
                self.pull_status = Some(format!("LM Studio download failed: {}", e));
            }
        }
    }

    /// Poll the active pull for progress. Called each TUI tick.
    pub fn tick_pull(&mut self) {
        self.enqueue_capability_probes_for_visible(24);
        self.tick_download_capability();
        self.tick_catalog_refresh();
        if self.pull_active.is_some() {
            self.tick_count = self.tick_count.wrapping_add(1);
        }
        let Some(handle) = &self.pull_active else {
            return;
        };
        // Drain all available events
        loop {
            match handle.receiver.try_recv() {
                Ok(PullEvent::Progress { status, percent }) => {
                    self.pull_percent = percent;
                    self.pull_status = Some(status);
                }
                Ok(PullEvent::Done) => {
                    let completed_provider = self.pull_provider;
                    let completed_model_name = self.pull_model_name.clone();
                    self.pull_percent = None;
                    self.pull_active = None;
                    self.pull_provider = None;
                    self.refresh_installed();

                    let done_msg = if completed_provider == Some(ActivePullProvider::LmStudio)
                        && let Some(model_name) = completed_model_name.as_deref()
                        && !providers::is_model_installed_lmstudio(
                            model_name,
                            &self.lmstudio_installed,
                        )
                    {
                        format!(
                            "LM Studio reported completion, but '{}' is not installed yet",
                            model_name
                        )
                    } else if let Some(provider) = completed_provider {
                        format!("Download complete via {}!", provider.label())
                    } else {
                        "Download complete!".to_string()
                    };
                    self.pull_status = Some(done_msg);
                    return;
                }
                Ok(PullEvent::Error(e)) => {
                    self.pull_status = Some(format!("Error: {}", e));
                    self.pull_percent = None;
                    self.pull_active = None;
                    self.pull_provider = None;
                    return;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.pull_status = Some("Pull ended".to_string());
                    self.pull_percent = None;
                    self.pull_active = None;
                    self.pull_provider = None;
                    self.refresh_installed();
                    return;
                }
            }
        }
    }

    fn available_download_providers(
        &self,
        model_name: &str,
        has_catalog_gguf: bool,
    ) -> Vec<DownloadProvider> {
        let mut providers_for_model = Vec::new();
        if providers::has_ollama_mapping(model_name)
            && (self.ollama_available || self.ollama_binary_available)
        {
            providers_for_model.push(DownloadProvider::Ollama);
        }
        if self.mlx_available {
            providers_for_model.push(DownloadProvider::Mlx);
        }
        // Check catalog gguf_sources first (no HTTP probe needed), then
        // fall back to the heuristic repo lookup
        if self.llamacpp_available
            && (has_catalog_gguf || providers::first_existing_gguf_repo(model_name).is_some())
        {
            providers_for_model.push(DownloadProvider::LlamaCpp);
        }
        if self.docker_mr_available && providers::has_docker_mr_mapping(model_name) {
            providers_for_model.push(DownloadProvider::DockerModelRunner);
        }
        if self.lmstudio_available && providers::has_lmstudio_mapping(model_name) {
            providers_for_model.push(DownloadProvider::LmStudio);
        }
        providers_for_model
    }

    fn open_download_provider_popup(&mut self, model_name: String, options: Vec<DownloadProvider>) {
        self.download_provider_model = Some(model_name);
        self.download_provider_options = options;
        self.download_provider_cursor = 0;
        self.input_mode = InputMode::DownloadProviderPopup;
        self.pull_status = Some("Choose download runtime and press Enter".to_string());
    }

    pub fn close_download_provider_popup(&mut self) {
        self.download_provider_model = None;
        self.download_provider_options.clear();
        self.download_provider_cursor = 0;
        self.input_mode = InputMode::Normal;
        self.pull_status = Some("Download cancelled".to_string());
    }

    pub fn download_provider_popup_up(&mut self) {
        if self.download_provider_cursor > 0 {
            self.download_provider_cursor -= 1;
        }
    }

    pub fn download_provider_popup_down(&mut self) {
        if self.download_provider_cursor + 1 < self.download_provider_options.len() {
            self.download_provider_cursor += 1;
        }
    }

    pub fn confirm_download_provider_selection(&mut self) {
        let Some(model_name) = self.download_provider_model.clone() else {
            self.input_mode = InputMode::Normal;
            return;
        };
        let Some(provider) = self
            .download_provider_options
            .get(self.download_provider_cursor)
            .copied()
        else {
            self.close_download_provider_popup();
            return;
        };

        self.download_provider_model = None;
        self.download_provider_options.clear();
        self.download_provider_cursor = 0;
        self.input_mode = InputMode::Normal;
        self.start_download_with_provider(model_name, provider);
    }

    /// Re-query all providers for installed models and update all_fits.
    pub fn refresh_installed(&mut self) {
        let (ollama_set, ollama_count) = self.ollama.installed_models_counted();
        self.ollama_installed = ollama_set;
        self.ollama_installed_count = ollama_count;
        self.mlx_installed = self.mlx.installed_models();
        let (llamacpp_set, llamacpp_count) = self.llamacpp.installed_models_counted();
        self.llamacpp_installed = llamacpp_set;
        self.llamacpp_installed_count = llamacpp_count;
        let (docker_mr_set, docker_mr_count) = self.docker_mr.installed_models_counted();
        self.docker_mr_installed = docker_mr_set;
        self.docker_mr_installed_count = docker_mr_count;
        let (lmstudio_set, lmstudio_count) = self.lmstudio.installed_models_counted();
        self.lmstudio_installed = lmstudio_set;
        self.lmstudio_installed_count = lmstudio_count;
        let (vllm_available, vllm_set, vllm_count) = self.vllm.detect_with_installed();
        self.vllm_available = vllm_available;
        self.vllm_installed = vllm_set;
        self.vllm_installed_count = vllm_count;
        self.rebuild_fits();
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn download_capability_for(&self, model_name: &str) -> DownloadCapability {
        self.download_capabilities
            .get(model_name)
            .copied()
            .unwrap_or(DownloadCapability::Unknown)
    }

    pub fn enqueue_capability_probes_for_visible(&mut self, window: usize) {
        if self.filtered_fits.is_empty() {
            return;
        }
        let start = self.selected_row.saturating_sub(window / 2);
        let end = (start + window).min(self.filtered_fits.len());
        for idx in start..end {
            if let Some(&fit_idx) = self.filtered_fits.get(idx) {
                let model_name = self.all_fits[fit_idx].model.name.clone();
                let has_catalog_gguf = !self.all_fits[fit_idx].model.gguf_sources.is_empty();
                self.enqueue_capability_probe(model_name, has_catalog_gguf);
            }
        }
    }

    fn enqueue_capability_probe(&mut self, model_name: String, has_catalog_gguf: bool) {
        if self.download_capabilities.contains_key(&model_name)
            || self.download_capability_inflight.contains(&model_name)
            || self.download_capability_inflight.len() >= 12
        {
            return;
        }
        self.download_capability_inflight.insert(model_name.clone());

        let tx = self.download_capability_tx.clone();
        let ollama_runtime_available = self.ollama_available || self.ollama_binary_available;
        let llamacpp_available = self.llamacpp_available;
        let docker_mr_available = self.docker_mr_available;
        let lmstudio_available = self.lmstudio_available;
        std::thread::spawn(move || {
            let has_ollama = ollama_runtime_available && providers::has_ollama_mapping(&model_name);
            let has_llamacpp = if llamacpp_available {
                // Use catalog data when available to skip slow HTTP probes
                has_catalog_gguf || providers::first_existing_gguf_repo(&model_name).is_some()
            } else {
                false
            };
            let has_docker = docker_mr_available && providers::has_docker_mr_mapping(&model_name);
            let has_lmstudio = lmstudio_available && providers::has_lmstudio_mapping(&model_name);

            let mut flags = 0u8;
            if has_ollama {
                flags |= DL_OLLAMA;
            }
            if has_llamacpp {
                flags |= DL_LLAMACPP;
            }
            if has_docker {
                flags |= DL_DOCKER;
            }
            if has_lmstudio {
                flags |= DL_LMSTUDIO;
            }
            let _ = tx.send((model_name, DownloadCapability::Known(flags)));
        });
    }

    fn tick_download_capability(&mut self) {
        loop {
            match self.download_capability_rx.try_recv() {
                Ok((name, capability)) => {
                    self.download_capability_inflight.remove(&name);
                    self.download_capabilities.insert(name, capability);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
    }

    fn active_plan_input(&self) -> &String {
        match self.plan_field {
            PlanField::Context => &self.plan_context_input,
            PlanField::Quant => &self.plan_quant_input,
            PlanField::TargetTps => &self.plan_target_tps_input,
        }
    }

    fn active_plan_input_mut(&mut self) -> &mut String {
        match self.plan_field {
            PlanField::Context => &mut self.plan_context_input,
            PlanField::Quant => &mut self.plan_quant_input,
            PlanField::TargetTps => &mut self.plan_target_tps_input,
        }
    }
}

fn command_exists(name: &str) -> bool {
    std::process::Command::new("which")
        .arg(name)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::{App, AvailabilityFilter, ContextFilter, FilterState, FitFilter, RuntimeFilter, TpFilter};
    use llmfit_core::{
        fit::{FitLevel, InferenceRuntime, ModelFit, RunMode, ScoreComponents},
        hardware::{GpuBackend, SystemSpecs},
        models::{Capability, GgufSource, LlmModel, ModelFormat, UseCase},
    };
    use std::collections::{HashMap, HashSet, VecDeque};
    use std::fs;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("llmfit-{name}-{nanos}.json"))
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = temp_path(name);
        fs::create_dir_all(&dir).expect("should create temp dir");
        dir
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn take_env_lock() -> std::sync::MutexGuard<'static, ()> {
        match env_lock().lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    struct EnvGuard {
        old_host: Option<String>,
        old_path: Option<String>,
    }

    impl EnvGuard {
        fn install(host: Option<&str>, prepend_path: Option<&std::path::Path>) -> Self {
            let old_host = std::env::var("LMSTUDIO_HOST").ok();
            let old_path = std::env::var("PATH").ok();

            if let Some(host) = host {
                unsafe { std::env::set_var("LMSTUDIO_HOST", host) };
            } else {
                unsafe { std::env::remove_var("LMSTUDIO_HOST") };
            }

            if let Some(path) = prepend_path {
                let mut parts = vec![path.display().to_string()];
                if let Some(existing) = &old_path {
                    parts.push(existing.clone());
                }
                unsafe { std::env::set_var("PATH", parts.join(":")) };
            }

            Self { old_host, old_path }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(host) = &self.old_host {
                unsafe { std::env::set_var("LMSTUDIO_HOST", host) };
            } else {
                unsafe { std::env::remove_var("LMSTUDIO_HOST") };
            }

            if let Some(path) = &self.old_path {
                unsafe { std::env::set_var("PATH", path) };
            } else {
                unsafe { std::env::remove_var("PATH") };
            }
        }
    }

    #[derive(Clone)]
    struct MockHttpResponse {
        status: u16,
        body: String,
        content_type: &'static str,
    }

    impl MockHttpResponse {
        fn json(status: u16, body: &str) -> Self {
            Self {
                status,
                body: body.to_string(),
                content_type: "application/json",
            }
        }

        fn text(status: u16, body: &str) -> Self {
            Self {
                status,
                body: body.to_string(),
                content_type: "text/plain",
            }
        }
    }

    #[derive(Clone, Debug)]
    struct RecordedRequest {
        method: String,
        path: String,
        body: String,
    }

    struct MockLmStudioServer {
        addr: std::net::SocketAddr,
        requests: Arc<Mutex<Vec<RecordedRequest>>>,
        stop_tx: Option<mpsc::Sender<()>>,
        handle: Option<thread::JoinHandle<()>>,
    }

    impl MockLmStudioServer {
        fn start(routes: Vec<(String, Vec<MockHttpResponse>)>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").expect("should bind test server");
            listener
                .set_nonblocking(true)
                .expect("should set nonblocking");
            let addr = listener.local_addr().expect("should have local addr");
            let requests = Arc::new(Mutex::new(Vec::new()));
            let routes = Arc::new(Mutex::new(
                routes
                    .into_iter()
                    .map(|(path, responses)| (path, VecDeque::from(responses)))
                    .collect::<HashMap<_, _>>(),
            ));
            let requests_clone = Arc::clone(&requests);
            let routes_clone = Arc::clone(&routes);
            let (stop_tx, stop_rx) = mpsc::channel();

            let handle = thread::spawn(move || loop {
                if stop_rx.try_recv().is_ok() {
                    break;
                }

                match listener.accept() {
                    Ok((mut stream, _)) => {
                        if let Some(request) = read_mock_request(&mut stream) {
                            requests_clone
                                .lock()
                                .expect("requests lock")
                                .push(request.clone());
                            let key = format!("{} {}", request.method, request.path);
                            let response = {
                                let mut routes = routes_clone.lock().expect("routes lock");
                                if let Some(queue) = routes.get_mut(&key) {
                                    if queue.len() > 1 {
                                        queue.pop_front().expect("queued response")
                                    } else {
                                        queue.front().cloned().expect("sticky response")
                                    }
                                } else {
                                    MockHttpResponse::text(404, "not found")
                                }
                            };
                            write_mock_response(&mut stream, &response);
                        }
                    }
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(_) => break,
                }
            });

            Self {
                addr,
                requests,
                stop_tx: Some(stop_tx),
                handle: Some(handle),
            }
        }

        fn base_url(&self) -> String {
            format!("http://{}", self.addr)
        }

        fn requests(&self) -> Vec<RecordedRequest> {
            self.requests.lock().expect("requests lock").clone()
        }
    }

    impl Drop for MockLmStudioServer {
        fn drop(&mut self) {
            if let Some(stop_tx) = self.stop_tx.take() {
                let _ = stop_tx.send(());
            }
            if let Some(handle) = self.handle.take() {
                let _ = handle.join();
            }
        }
    }

    fn read_mock_request(stream: &mut TcpStream) -> Option<RecordedRequest> {
        let mut buffer = Vec::new();
        let mut temp = [0_u8; 1024];
        let header_end;
        loop {
            let read = stream.read(&mut temp).ok()?;
            if read == 0 {
                return None;
            }
            buffer.extend_from_slice(&temp[..read]);
            if let Some(pos) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
                header_end = pos + 4;
                break;
            }
        }

        let header_text = String::from_utf8_lossy(&buffer[..header_end]);
        let mut lines = header_text.lines();
        let request_line = lines.next()?.trim();
        let mut parts = request_line.split_whitespace();
        let method = parts.next()?.to_string();
        let path = parts.next()?.to_string();

        let content_length = header_text
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.eq_ignore_ascii_case("content-length") {
                    value.trim().parse::<usize>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(0);

        while buffer.len() < header_end + content_length {
            let read = stream.read(&mut temp).ok()?;
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&temp[..read]);
        }

        let body = String::from_utf8_lossy(&buffer[header_end..header_end + content_length]).into();
        Some(RecordedRequest { method, path, body })
    }

    fn write_mock_response(stream: &mut TcpStream, response: &MockHttpResponse) {
        let status_text = match response.status {
            200 => "OK",
            404 => "Not Found",
            500 => "Internal Server Error",
            _ => "OK",
        };
        let raw = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response.status,
            status_text,
            response.content_type,
            response.body.len(),
            response.body,
        );
        stream
            .write_all(raw.as_bytes())
            .expect("should write mock response");
        let _ = stream.flush();
    }

    struct FakeLmsCli {
        dir: PathBuf,
        log_path: PathBuf,
    }

    impl FakeLmsCli {
        fn install(name: &str, server_start_success: bool, get_success: bool) -> Self {
            Self::install_with_scripts(
                name,
                server_start_success,
                if get_success {
                    "exit 0"
                } else {
                    "echo 'cli get failed' >&2\nexit 1"
                },
                "[]",
            )
        }

        fn install_with_get_script(
            name: &str,
            server_start_success: bool,
            get_script: &str,
        ) -> Self {
            Self::install_with_scripts(name, server_start_success, get_script, "[]")
        }

        fn install_with_scripts(
            name: &str,
            server_start_success: bool,
            get_script: &str,
            ls_json: &str,
        ) -> Self {
            let dir = temp_dir(name);
            let log_path = dir.join("lms.log");
            let script_path = dir.join("lms");
            let server_block = if server_start_success {
                "exit 0"
            } else {
                "echo 'server start failed' >&2\nexit 1"
            };
            let script = format!(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"{}\"\nif [ \"$1\" = \"ls\" ]; then\ncat <<'EOF'\n{}\nEOF\nexit 0\nfi\nif [ \"$1\" = \"server\" ] && [ \"$2\" = \"start\" ]; then\n{}\nfi\nif [ \"$1\" = \"get\" ]; then\n{}\nfi\necho 'unsupported args' >&2\nexit 1\n",
                log_path.display(),
                ls_json,
                server_block,
                get_script,
            );
            fs::write(&script_path, script).expect("should write fake lms script");
            let mut perms = fs::metadata(&script_path)
                .expect("script metadata")
                .permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script_path, perms).expect("set script perms");
            Self { dir, log_path }
        }

        fn logged_commands(&self) -> Vec<String> {
            match fs::read_to_string(&self.log_path) {
                Ok(raw) => raw.lines().map(|line| line.to_string()).collect(),
                Err(_) => Vec::new(),
            }
        }
    }

    fn test_specs() -> SystemSpecs {
        SystemSpecs {
            total_ram_gb: 64.0,
            available_ram_gb: 48.0,
            total_cpu_cores: 16,
            cpu_name: "Test CPU".to_string(),
            has_gpu: true,
            gpu_vram_gb: Some(24.0),
            total_gpu_vram_gb: Some(24.0),
            gpu_name: Some("RTX 4090".to_string()),
            gpu_count: 1,
            unified_memory: false,
            backend: GpuBackend::Cuda,
            gpus: vec![],
            cluster_mode: false,
            cluster_node_count: 0,
        }
    }

    fn test_app_with_model(model_name: &str, gguf_sources: Vec<GgufSource>) -> App {
        let (download_capability_tx, download_capability_rx) = mpsc::channel();
        let (catalog_refresh_tx, catalog_refresh_rx) = mpsc::channel();
        App {
            should_quit: false,
            input_mode: super::InputMode::Normal,
            search_query: String::new(),
            cursor_position: 0,
            specs: test_specs(),
            source_models: vec![],
            base_context_limit: None,
            all_fits: vec![ModelFit {
                model: LlmModel {
                    name: model_name.to_string(),
                    provider: "huggingface".to_string(),
                    parameter_count: "3B".to_string(),
                    parameters_raw: Some(3_000_000_000),
                    min_ram_gb: 2.0,
                    recommended_ram_gb: 4.0,
                    min_vram_gb: Some(2.0),
                    quantization: "Q4_K_M".to_string(),
                    context_length: 32_768,
                    use_case: "general".to_string(),
                    is_moe: false,
                    num_experts: None,
                    active_experts: None,
                    active_parameters: None,
                    release_date: None,
                    gguf_sources,
                    capabilities: vec![],
                    format: ModelFormat::Gguf,
                    num_attention_heads: None,
                    num_key_value_heads: None,
                    metadata_overlay: None,
                },
                fit_level: FitLevel::Good,
                run_mode: RunMode::Gpu,
                memory_required_gb: 2.0,
                memory_available_gb: 24.0,
                utilization_pct: 10.0,
                notes: vec![],
                moe_offloaded_gb: None,
                score: 90.0,
                score_components: ScoreComponents {
                    quality: 90.0,
                    speed: 80.0,
                    fit: 95.0,
                    context: 85.0,
                },
                estimated_tps: 42.0,
                best_quant: "Q4_K_M".to_string(),
                use_case: UseCase::General,
                runtime: InferenceRuntime::LlamaCpp,
                installed: false,
            }],
            filtered_fits: vec![],
            providers: vec![],
            selected_providers: vec![],
            use_cases: vec![],
            selected_use_cases: vec![],
            capabilities: vec![],
            selected_capabilities: vec![],
            fit_filter: super::FitFilter::All,
            runtime_filter: RuntimeFilter::Any,
            availability_filter: super::AvailabilityFilter::All,
            tp_filter: super::TpFilter::All,
            context_filter: super::ContextFilter::All,
            installed_first: false,
            sort_column: llmfit_core::fit::SortColumn::Score,
            sort_ascending: false,
            selected_row: 0,
            show_detail: false,
            show_compare: false,
            compare_mark_model: None,
            show_multi_compare: false,
            compare_models: vec![],
            compare_scroll: 0,
            show_plan: false,
            plan_model_idx: None,
            plan_field: super::PlanField::Context,
            plan_context_input: String::new(),
            plan_quant_input: String::new(),
            plan_target_tps_input: String::new(),
            plan_cursor_position: 0,
            plan_estimate: None,
            plan_error: None,
            provider_cursor: 0,
            use_case_cursor: 0,
            capability_cursor: 0,
            download_provider_cursor: 0,
            download_provider_options: vec![],
            download_provider_model: None,
            ollama_available: false,
            ollama_binary_available: false,
            ollama_installed: HashSet::new(),
            ollama_installed_count: 0,
            ollama: llmfit_core::providers::OllamaProvider::new(),
            mlx_available: false,
            mlx_installed: HashSet::new(),
            mlx: llmfit_core::providers::MlxProvider::new(),
            llamacpp_available: false,
            llamacpp_installed: HashSet::new(),
            llamacpp_installed_count: 0,
            llamacpp_detection_hint: String::new(),
            llamacpp: llmfit_core::providers::LlamaCppProvider::new(),
            docker_mr_available: false,
            docker_mr_installed: HashSet::new(),
            docker_mr_installed_count: 0,
            docker_mr: llmfit_core::providers::DockerModelRunnerProvider::new(),
            lmstudio_available: true,
            lmstudio_installed: HashSet::new(),
            lmstudio_installed_count: 0,
            lmstudio: llmfit_core::providers::LmStudioProvider::new(),
            vllm_available: false,
            vllm_installed: HashSet::new(),
            vllm_installed_count: 0,
            vllm: llmfit_core::providers::VllmProvider::new(),
            pull_active: None,
            pull_status: None,
            pull_percent: None,
            pull_model_name: None,
            pull_provider: None,
            download_capabilities: HashMap::new(),
            download_capability_inflight: HashSet::new(),
            download_capability_tx,
            download_capability_rx,
            catalog_refresh_active: false,
            catalog_refresh_tx,
            catalog_refresh_rx,
            tick_count: 0,
            confirm_download: false,
            visual_anchor: None,
            select_column: 2,
            quants: vec![],
            selected_quants: vec![],
            quant_cursor: 0,
            run_modes: vec![],
            selected_run_modes: vec![],
            run_mode_cursor: 0,
            params_buckets: vec![],
            selected_params_buckets: vec![],
            params_bucket_cursor: 0,
            theme: crate::theme::Theme::Default,
            backend_hidden_count: 0,
        }
    }

    fn wait_for_pull_completion(app: &mut App) -> String {
        let start = Instant::now();
        loop {
            app.tick_pull();
            if app.pull_active.is_none() {
                return app.pull_status.clone().unwrap_or_default();
            }
            assert!(
                start.elapsed() < Duration::from_secs(5),
                "timed out waiting for pull completion: {:?}",
                app.pull_status
            );
            thread::sleep(Duration::from_millis(25));
        }
    }

    #[test]
    fn lmstudio_download_picker_opens_from_tui_flow() {
        let _env_lock = take_env_lock();
        let server = MockLmStudioServer::start(vec![(
            "GET /v1/models".to_string(),
            vec![MockHttpResponse::json(200, r#"{"models":[]}"#)],
        )]);
        let _env = EnvGuard::install(Some(&server.base_url()), None);

        let mut app = test_app_with_model(
            "HuggingFaceTB/SmolLM3-3B",
            vec![GgufSource {
                repo: "lmstudio-community/SmolLM3-3B-GGUF".to_string(),
                provider: "lmstudio-community".to_string(),
            }],
        );
        app.filtered_fits = vec![0];

        app.start_download();

        assert_eq!(app.input_mode, super::InputMode::DownloadProviderPopup);
        assert_eq!(app.download_provider_options, vec![super::DownloadProvider::LmStudio]);
        assert_eq!(
            app.download_provider_model.as_deref(),
            Some("HuggingFaceTB/SmolLM3-3B")
        );
    }

    #[test]
    fn lmstudio_tui_flow_download_succeeds_via_http() {
        let _env_lock = take_env_lock();
        let server = MockLmStudioServer::start(vec![
            (
                "GET /v1/models".to_string(),
                vec![
                    MockHttpResponse::json(200, r#"{"models":[]}"#),
                    MockHttpResponse::json(
                        200,
                        r#"{"models":[{"key":"lmstudio-community/smollm3-3b-gguf"}]}"#,
                    ),
                ],
            ),
            (
                "POST /api/v1/models/download".to_string(),
                vec![MockHttpResponse::json(
                    200,
                    r#"{"status":"downloading","job_id":"job-1","total_size_bytes":1000000,"started_at":"2025-01-01T00:00:00Z"}"#,
                )],
            ),
            (
                "GET /api/v1/models/download/status/job-1".to_string(),
                vec![
                    MockHttpResponse::json(
                        200,
                        r#"{"job_id":"job-1","status":"downloading","total_size_bytes":1000000,"downloaded_bytes":500000}"#,
                    ),
                    MockHttpResponse::json(
                        200,
                        r#"{"job_id":"job-1","status":"completed","total_size_bytes":1000000,"downloaded_bytes":1000000,"completed_at":"2025-01-01T00:01:00Z"}"#,
                    ),
                ],
            ),
        ]);
        let _env = EnvGuard::install(Some(&server.base_url()), None);

        let mut app = test_app_with_model(
            "HuggingFaceTB/SmolLM3-3B",
            vec![GgufSource {
                repo: "lmstudio-community/SmolLM3-3B-GGUF".to_string(),
                provider: "lmstudio-community".to_string(),
            }],
        );

        app.start_lmstudio_download("HuggingFaceTB/SmolLM3-3B".to_string());
        let final_status = wait_for_pull_completion(&mut app);

        assert_eq!(final_status, "Download complete via LM Studio!");
        assert!(app.lmstudio_available);
        let requests = server.requests();
        assert!(requests.iter().any(|req| req.method == "POST" && req.path == "/api/v1/models/download" && req.body.contains("lmstudio-community/SmolLM3-3B-GGUF")));
        assert!(requests.iter().any(|req| req.method == "GET" && req.path == "/api/v1/models/download/status/job-1"));
    }

    #[test]
    fn tick_pull_clears_seeded_lmstudio_zero_percent_when_progress_is_unknown() {
        let mut app = test_app_with_model("smollm3-3b-gguf", vec![]);
        let (tx, rx) = mpsc::channel();
        app.pull_active = Some(llmfit_core::providers::PullHandle {
            model_tag: "smollm3-3b-gguf".to_string(),
            receiver: rx,
        });
        app.pull_percent = Some(0.0);
        app.pull_status = Some("Downloading smollm3-3b-gguf via LM Studio (downloading)".to_string());

        tx.send(llmfit_core::providers::PullEvent::Progress {
            status: "Downloading smollm3-3b-gguf via LM Studio (waiting for progress...)"
                .to_string(),
            percent: None,
        })
        .expect("should send progress event");

        app.tick_pull();

        assert_eq!(app.pull_percent, None);
        assert_eq!(
            app.pull_status.as_deref(),
            Some("Downloading smollm3-3b-gguf via LM Studio (waiting for progress...)")
        );
    }

    #[test]
    fn lmstudio_tui_flow_handles_already_downloaded() {
        let _env_lock = take_env_lock();
        let server = MockLmStudioServer::start(vec![
            (
                "GET /v1/models".to_string(),
                vec![
                    MockHttpResponse::json(200, r#"{"models":[]}"#),
                    MockHttpResponse::json(200, r#"{"models":[{"key":"smollm3-3b-gguf"}]}"#),
                ],
            ),
            (
                "POST /api/v1/models/download".to_string(),
                vec![MockHttpResponse::json(200, r#"{"status":"already_downloaded"}"#)],
            ),
        ]);
        let _env = EnvGuard::install(Some(&server.base_url()), None);

        let mut app = test_app_with_model("smollm3-3b-gguf", vec![]);

        app.start_lmstudio_download("smollm3-3b-gguf".to_string());
        let final_status = wait_for_pull_completion(&mut app);

        assert_eq!(final_status, "Download complete via LM Studio!");
        assert!(app.lmstudio_installed.contains("smollm3-3b-gguf"));
    }

    #[test]
    fn lmstudio_tui_flow_falls_back_to_cli_after_http_404() {
        let _env_lock = take_env_lock();
        let fake_cli = FakeLmsCli::install("lmstudio-cli-success", true, true);
        let server = MockLmStudioServer::start(vec![
            (
                "GET /v1/models".to_string(),
                vec![
                    MockHttpResponse::json(200, r#"{"models":[]}"#),
                    MockHttpResponse::json(
                        200,
                        r#"{"models":[{"key":"lmstudio-community/smollm3-3b-gguf"}]}"#,
                    ),
                ],
            ),
            (
                "POST /api/v1/models/download".to_string(),
                vec![MockHttpResponse::text(404, "missing")],
            ),
        ]);
        let _env = EnvGuard::install(Some(&server.base_url()), Some(&fake_cli.dir));

        let mut app = test_app_with_model(
            "HuggingFaceTB/SmolLM3-3B",
            vec![GgufSource {
                repo: "lmstudio-community/SmolLM3-3B-GGUF".to_string(),
                provider: "lmstudio-community".to_string(),
            }],
        );

        app.start_lmstudio_download("HuggingFaceTB/SmolLM3-3B".to_string());
        let final_status = wait_for_pull_completion(&mut app);

        assert_eq!(final_status, "Download complete via LM Studio!");
        let commands = fake_cli.logged_commands();
        assert!(commands.iter().any(|cmd| cmd == "get SmolLM3-3B-GGUF --yes"));
    }

    #[test]
    fn lmstudio_tui_flow_reports_cli_failure_after_http_404() {
        let _env_lock = take_env_lock();
        let fake_cli = FakeLmsCli::install("lmstudio-cli-fail", true, false);
        let server = MockLmStudioServer::start(vec![
            (
                "GET /v1/models".to_string(),
                vec![MockHttpResponse::json(200, r#"{"models":[]}"#)],
            ),
            (
                "POST /api/v1/models/download".to_string(),
                vec![MockHttpResponse::text(404, "missing")],
            ),
        ]);
        let _env = EnvGuard::install(Some(&server.base_url()), Some(&fake_cli.dir));

        let mut app = test_app_with_model(
            "HuggingFaceTB/SmolLM3-3B",
            vec![GgufSource {
                repo: "lmstudio-community/SmolLM3-3B-GGUF".to_string(),
                provider: "lmstudio-community".to_string(),
            }],
        );

        app.start_lmstudio_download("HuggingFaceTB/SmolLM3-3B".to_string());
        let final_status = wait_for_pull_completion(&mut app);

        assert!(final_status.contains("CLI fallback failed"));
        let commands = fake_cli.logged_commands();
        assert!(commands.iter().any(|cmd| cmd == "get SmolLM3-3B-GGUF --yes"));
    }

    #[test]
    fn lmstudio_tui_flow_uses_direct_cli_when_api_unavailable_for_native_key() {
        let _env_lock = take_env_lock();
        let fake_cli = FakeLmsCli::install_with_scripts(
            "lmstudio-cli-direct",
            false,
            "exit 0",
            r#"[{"modelKey":"smollm3-3b-gguf"}]"#,
        );
        let _env = EnvGuard::install(Some("http://127.0.0.1:9"), Some(&fake_cli.dir));

        let mut app = test_app_with_model("smollm3-3b-gguf", vec![]);

        app.start_lmstudio_download("smollm3-3b-gguf".to_string());
        let final_status = wait_for_pull_completion(&mut app);

        assert_eq!(final_status, "Download complete via LM Studio!");
        let commands = fake_cli.logged_commands();
        assert!(commands.iter().any(|cmd| cmd == "server start"));
        assert!(commands.iter().any(|cmd| cmd == "get smollm3-3b-gguf --yes"));
    }

    #[test]
    fn lmstudio_done_requires_installed_model_match_before_reporting_success() {
        let _env_lock = take_env_lock();
        let server = MockLmStudioServer::start(vec![
            (
                "GET /v1/models".to_string(),
                vec![
                    MockHttpResponse::json(200, r#"{"models":[]}"#),
                    MockHttpResponse::json(
                        200,
                        r#"{"models":[{"key":"lmstudio-community/smollm3-3b-gguf"}]}"#,
                    ),
                ],
            ),
            (
                "POST /api/v1/models/download".to_string(),
                vec![MockHttpResponse::json(200, r#"{"status":"already_downloaded"}"#)],
            ),
        ]);
        let _env = EnvGuard::install(Some(&server.base_url()), None);

        let mut app = test_app_with_model("microsoft/Phi-4-reasoning", vec![]);

        app.start_lmstudio_download("microsoft/Phi-4-reasoning".to_string());
        let final_status = wait_for_pull_completion(&mut app);

        assert_eq!(
            final_status,
            "LM Studio reported completion, but 'microsoft/Phi-4-reasoning' is not installed yet"
        );
    }

    #[test]
    fn lmstudio_cli_fallback_emits_live_activity_updates() {
        let _env_lock = take_env_lock();
        let fake_cli = FakeLmsCli::install_with_scripts(
            "lmstudio-cli-progress",
            false,
            "printf 'Resolving model\\n'; sleep 0.2; printf 'Downloading shard 1/3\\n'; sleep 0.2; printf 'Downloading shard 2/3\\n'; sleep 0.2; exit 0",
            r#"[{"modelKey":"smollm3-3b-gguf"}]"#,
        );
        let _env = EnvGuard::install(Some("http://127.0.0.1:9"), Some(&fake_cli.dir));

        let mut app = test_app_with_model("smollm3-3b-gguf", vec![]);

        app.start_lmstudio_download("smollm3-3b-gguf".to_string());

        let start = Instant::now();
        let mut saw_cli_activity = false;
        while start.elapsed() < Duration::from_secs(3) {
            app.tick_pull();
            if app
                .pull_status
                .as_deref()
                .is_some_and(|status| status.contains("LM Studio CLI: Downloading shard 1/3"))
            {
                saw_cli_activity = true;
                break;
            }
            thread::sleep(Duration::from_millis(25));
        }

        assert!(saw_cli_activity, "expected streamed CLI activity update");
        let final_status = wait_for_pull_completion(&mut app);
        assert!(
            final_status == "Download complete via LM Studio!"
                || final_status
                    == "LM Studio reported completion, but 'HuggingFaceTB/SmolLM3-3B' is not installed yet"
        );
    }

    #[test]
    fn lmstudio_cli_fallback_emits_numeric_progress_from_carriage_return_updates() {
        let _env_lock = take_env_lock();
        let fake_cli = FakeLmsCli::install_with_scripts(
            "lmstudio-cli-percent-progress",
            true,
            "i=0; while [ $i -lt 6 ]; do printf '\r⠋ [████████████▌         ]  57.23%% |  2.67 GB /  4.68 GB |  45.12 MB/s | ETA 00:43\r'; sleep 0.2; i=$((i+1)); done; printf '⠙ [██████████████████████] 100.00%% |  4.68 GB /  4.68 GB |  51.00 MB/s | ETA 00:00\n'; exit 0",
            r#"[{"modelKey":"smollm3-3b-gguf"}]"#,
        );
        let server = MockLmStudioServer::start(vec![
            (
                "GET /v1/models".to_string(),
                vec![MockHttpResponse::json(200, r#"{"models":[]}"#)],
            ),
            (
                "POST /api/v1/models/download".to_string(),
                vec![MockHttpResponse::text(404, "missing")],
            ),
        ]);
        let _env = EnvGuard::install(Some(&server.base_url()), Some(&fake_cli.dir));

        let mut app = test_app_with_model(
            "HuggingFaceTB/SmolLM3-3B",
            vec![GgufSource {
                repo: "lmstudio-community/SmolLM3-3B-GGUF".to_string(),
                provider: "lmstudio-community".to_string(),
            }],
        );

        app.start_lmstudio_download("HuggingFaceTB/SmolLM3-3B".to_string());

        let start = Instant::now();
        let mut saw_numeric_progress = false;
        while start.elapsed() < Duration::from_secs(5) {
            let Some(handle) = app.pull_active.as_ref() else {
                break;
            };

            match handle.receiver.try_recv() {
                Ok(llmfit_core::providers::PullEvent::Progress { status, percent }) => {
                    if percent.is_some_and(|pct| (pct - 57.23).abs() < 0.01)
                        && status.contains("57.23%")
                    {
                        saw_numeric_progress = true;
                        break;
                    }
                }
                Ok(_) => {}
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
            thread::sleep(Duration::from_millis(25));
        }

        assert!(
            saw_numeric_progress,
            "expected streamed CLI percent update from carriage-return progress bar"
        );
    }

    #[test]
    fn tick_pull_applies_numeric_progress_event() {
        let mut app = test_app_with_model("smollm3-3b-gguf", vec![]);
        let (tx, rx) = mpsc::channel();
        app.pull_active = Some(llmfit_core::providers::PullHandle {
            model_tag: "smollm3-3b-gguf".to_string(),
            receiver: rx,
        });

        tx.send(llmfit_core::providers::PullEvent::Progress {
            status: "LM Studio CLI: ⠋ [████████████▌         ]  57.23% |  2.67 GB /  4.68 GB |  45.12 MB/s | ETA 00:43".to_string(),
            percent: Some(57.23),
        })
        .expect("should send progress event");

        app.tick_pull();

        assert_eq!(app.pull_status.as_deref(), Some("LM Studio CLI: ⠋ [████████████▌         ]  57.23% |  2.67 GB /  4.68 GB |  45.12 MB/s | ETA 00:43"));
        assert!(app.pull_percent.is_some_and(|pct| (pct - 57.23).abs() < 0.01));
    }

    #[test]
    fn tick_pull_clears_stale_percent_when_progress_event_has_no_percent() {
        let mut app = test_app_with_model("smollm3-3b-gguf", vec![]);
        let (tx, rx) = mpsc::channel();
        app.pull_active = Some(llmfit_core::providers::PullHandle {
            model_tag: "smollm3-3b-gguf".to_string(),
            receiver: rx,
        });
        app.pull_percent = Some(0.0);
        app.pull_status = Some("Downloading via LM Studio (downloading)".to_string());

        tx.send(llmfit_core::providers::PullEvent::Progress {
            status: "LM Studio API unavailable (http status: 404); falling back to CLI (gemma-3n-E2B-it)".to_string(),
            percent: None,
        })
        .expect("should send progress event");

        app.tick_pull();

        assert_eq!(
            app.pull_status.as_deref(),
            Some("LM Studio API unavailable (http status: 404); falling back to CLI (gemma-3n-E2B-it)")
        );
        assert_eq!(app.pull_percent, None);
    }

    #[test]
    fn runtime_filter_cycles_through_all_values() {
        assert_eq!(RuntimeFilter::Any.next(), RuntimeFilter::LlamaCpp);
        assert_eq!(RuntimeFilter::LlamaCpp.next(), RuntimeFilter::Mlx);
        assert_eq!(RuntimeFilter::Mlx.next(), RuntimeFilter::Vllm);
        assert_eq!(RuntimeFilter::Vllm.next(), RuntimeFilter::LmStudio);
        assert_eq!(RuntimeFilter::LmStudio.next(), RuntimeFilter::Any);
    }

    #[test]
    fn filter_label_parsers_fallback_to_defaults() {
        assert_eq!(FitFilter::from_label("unknown"), FitFilter::All);
        assert_eq!(RuntimeFilter::from_label("unknown"), RuntimeFilter::Any);
        assert_eq!(AvailabilityFilter::from_label("unknown"), AvailabilityFilter::All);
        assert_eq!(TpFilter::from_label("unknown"), TpFilter::All);
        assert_eq!(ContextFilter::from_label("unknown"), ContextFilter::All);
        assert_eq!(App::parse_sort_column("unknown"), llmfit_core::fit::SortColumn::Score);
    }

    #[test]
    fn filter_state_loads_empty_json_with_defaults() {
        let path = temp_path("empty-filter-state");
        fs::write(&path, "{}").expect("should write test config file");

        let loaded = FilterState::load_from_path(&path).expect("should load empty config");

        assert_eq!(loaded.fit_filter, "All");
        assert_eq!(loaded.runtime_filter, "Any");
        assert_eq!(loaded.availability_filter, "All");
        assert_eq!(loaded.tp_filter, "All");
        assert_eq!(loaded.context_filter, "All");
        assert!(!loaded.installed_first);
        assert_eq!(loaded.sort_column, "Score");
        assert!(!loaded.sort_ascending);
        assert_eq!(loaded.selected_providers, None);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn filter_state_roundtrip_preserves_empty_and_named_selections() {
        let state = FilterState {
            fit_filter: "Runnable".to_string(),
            runtime_filter: "LM Studio".to_string(),
            availability_filter: "Installed".to_string(),
            tp_filter: "TP=4".to_string(),
            context_filter: ">=128k".to_string(),
            installed_first: true,
            sort_column: "Params".to_string(),
            sort_ascending: true,
            selected_providers: Some(vec![]),
            selected_use_cases: Some(vec!["Coding".to_string(), "Chat".to_string()]),
            selected_capabilities: Some(vec!["Tool Use".to_string()]),
            selected_quants: Some(vec!["Q4_K_M".to_string()]),
            selected_run_modes: Some(vec!["GPU".to_string(), "CPU".to_string()]),
            selected_params_buckets: Some(vec!["7-14B".to_string()]),
        };
        let path = temp_path("roundtrip-filter-state");

        state.save_to_path(&path);
        let loaded = FilterState::load_from_path(&path).expect("should load saved config");

        assert_eq!(loaded.fit_filter, state.fit_filter);
        assert_eq!(loaded.runtime_filter, state.runtime_filter);
        assert_eq!(loaded.availability_filter, state.availability_filter);
        assert_eq!(loaded.tp_filter, state.tp_filter);
        assert_eq!(loaded.context_filter, state.context_filter);
        assert_eq!(loaded.installed_first, state.installed_first);
        assert_eq!(loaded.sort_column, state.sort_column);
        assert_eq!(loaded.sort_ascending, state.sort_ascending);
        assert_eq!(loaded.selected_providers, state.selected_providers);
        assert_eq!(loaded.selected_use_cases, state.selected_use_cases);
        assert_eq!(loaded.selected_capabilities, state.selected_capabilities);
        assert_eq!(loaded.selected_quants, state.selected_quants);
        assert_eq!(loaded.selected_run_modes, state.selected_run_modes);
        assert_eq!(loaded.selected_params_buckets, state.selected_params_buckets);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn apply_filter_state_maps_saved_names_to_current_order() {
        let (download_capability_tx, download_capability_rx) = mpsc::channel();
        let (catalog_refresh_tx, catalog_refresh_rx) = mpsc::channel();
        let mut app = App {
            should_quit: false,
            input_mode: super::InputMode::Normal,
            search_query: String::new(),
            cursor_position: 0,
            specs: SystemSpecs {
                total_ram_gb: 64.0,
                available_ram_gb: 48.0,
                total_cpu_cores: 16,
                cpu_name: "Test CPU".to_string(),
                has_gpu: true,
                gpu_vram_gb: Some(24.0),
                total_gpu_vram_gb: Some(24.0),
                gpu_name: Some("RTX 4090".to_string()),
                gpu_count: 1,
                unified_memory: false,
                backend: GpuBackend::Cuda,
                gpus: vec![],
                cluster_mode: false,
                cluster_node_count: 0,
            },
            source_models: vec![],
            base_context_limit: None,
            all_fits: vec![],
            filtered_fits: vec![],
            providers: vec!["zeta".to_string(), "alpha".to_string(), "meta".to_string()],
            selected_providers: vec![true, true, true],
            use_cases: vec![UseCase::General, UseCase::Coding, UseCase::Chat],
            selected_use_cases: vec![true, true, true],
            capabilities: vec![Capability::Vision, Capability::ToolUse],
            selected_capabilities: vec![true, true],
            fit_filter: FitFilter::All,
            runtime_filter: RuntimeFilter::Any,
            availability_filter: AvailabilityFilter::All,
            tp_filter: TpFilter::All,
            context_filter: ContextFilter::All,
            installed_first: false,
            sort_column: llmfit_core::fit::SortColumn::Score,
            sort_ascending: false,
            selected_row: 0,
            show_detail: false,
            show_compare: false,
            compare_mark_model: None,
            show_multi_compare: false,
            compare_models: vec![],
            compare_scroll: 0,
            show_plan: false,
            plan_model_idx: None,
            plan_field: super::PlanField::Context,
            plan_context_input: String::new(),
            plan_quant_input: String::new(),
            plan_target_tps_input: String::new(),
            plan_cursor_position: 0,
            plan_estimate: None,
            plan_error: None,
            provider_cursor: 0,
            use_case_cursor: 0,
            capability_cursor: 0,
            download_provider_cursor: 0,
            download_provider_options: vec![],
            download_provider_model: None,
            ollama_available: false,
            ollama_binary_available: false,
            ollama_installed: HashSet::new(),
            ollama_installed_count: 0,
            ollama: llmfit_core::providers::OllamaProvider::new(),
            mlx_available: false,
            mlx_installed: HashSet::new(),
            mlx: llmfit_core::providers::MlxProvider::new(),
            llamacpp_available: false,
            llamacpp_installed: HashSet::new(),
            llamacpp_installed_count: 0,
            llamacpp_detection_hint: String::new(),
            llamacpp: llmfit_core::providers::LlamaCppProvider::new(),
            docker_mr_available: false,
            docker_mr_installed: HashSet::new(),
            docker_mr_installed_count: 0,
            docker_mr: llmfit_core::providers::DockerModelRunnerProvider::new(),
            lmstudio_available: false,
            lmstudio_installed: HashSet::new(),
            lmstudio_installed_count: 0,
            lmstudio: llmfit_core::providers::LmStudioProvider::new(),
            vllm_available: false,
            vllm_installed: HashSet::new(),
            vllm_installed_count: 0,
            vllm: llmfit_core::providers::VllmProvider::new(),
            pull_active: None,
            pull_status: None,
            pull_percent: None,
            pull_model_name: None,
            pull_provider: None,
            download_capabilities: HashMap::new(),
            download_capability_inflight: HashSet::new(),
            download_capability_tx,
            download_capability_rx,
            catalog_refresh_active: false,
            catalog_refresh_tx,
            catalog_refresh_rx,
            tick_count: 0,
            confirm_download: false,
            visual_anchor: None,
            select_column: 2,
            quants: vec!["Q8_0".to_string(), "Q4_K_M".to_string()],
            selected_quants: vec![true, true],
            quant_cursor: 0,
            run_modes: vec!["GPU".to_string(), "CPU+GPU".to_string(), "CPU".to_string()],
            selected_run_modes: vec![true, true, true],
            run_mode_cursor: 0,
            params_buckets: vec!["<3B".to_string(), "7-14B".to_string()],
            selected_params_buckets: vec![true, true],
            params_bucket_cursor: 0,
            theme: crate::theme::Theme::Default,
            backend_hidden_count: 0,
        };

        let state = FilterState {
            fit_filter: "Runnable".to_string(),
            runtime_filter: "vLLM".to_string(),
            availability_filter: "Installed".to_string(),
            tp_filter: "TP=3".to_string(),
            context_filter: ">=262k".to_string(),
            installed_first: true,
            sort_column: "Date".to_string(),
            sort_ascending: true,
            selected_providers: Some(vec!["meta".to_string()]),
            selected_use_cases: Some(vec!["Coding".to_string()]),
            selected_capabilities: Some(vec!["Tool Use".to_string()]),
            selected_quants: Some(vec!["Q4_K_M".to_string()]),
            selected_run_modes: Some(vec!["CPU".to_string()]),
            selected_params_buckets: Some(vec!["7-14B".to_string()]),
        };

        app.apply_filter_state(&state);

        assert_eq!(app.fit_filter, FitFilter::Runnable);
        assert_eq!(app.runtime_filter, RuntimeFilter::Vllm);
        assert_eq!(app.availability_filter, AvailabilityFilter::Installed);
        assert_eq!(app.tp_filter, TpFilter::Tp3);
        assert_eq!(app.context_filter, ContextFilter::AtLeast262k);
        assert!(app.installed_first);
        assert_eq!(app.sort_column, llmfit_core::fit::SortColumn::ReleaseDate);
        assert!(app.sort_ascending);
        assert_eq!(app.selected_providers, vec![false, false, true]);
        assert_eq!(app.selected_use_cases, vec![false, true, false]);
        assert_eq!(app.selected_capabilities, vec![false, true]);
        assert_eq!(app.selected_quants, vec![false, true]);
        assert_eq!(app.selected_run_modes, vec![false, false, true]);
        assert_eq!(app.selected_params_buckets, vec![false, true]);
    }

    #[test]
    fn reset_filters_restores_defaults() {
        let mut app = App::with_specs_and_context(
            SystemSpecs {
                total_ram_gb: 64.0,
                available_ram_gb: 48.0,
                total_cpu_cores: 16,
                cpu_name: "Test CPU".to_string(),
                has_gpu: true,
                gpu_vram_gb: Some(24.0),
                total_gpu_vram_gb: Some(24.0),
                gpu_name: Some("RTX 4090".to_string()),
                gpu_count: 1,
                unified_memory: false,
                backend: GpuBackend::Cuda,
                gpus: vec![],
                cluster_mode: false,
                cluster_node_count: 0,
            },
            None,
        );

        app.fit_filter = FitFilter::Runnable;
        app.runtime_filter = RuntimeFilter::LmStudio;
        app.availability_filter = AvailabilityFilter::Installed;
        app.tp_filter = TpFilter::Tp4;
        app.context_filter = ContextFilter::AtLeast128k;
        app.installed_first = true;
        app.sort_column = llmfit_core::fit::SortColumn::Params;
        app.sort_ascending = true;
        app.search_query = "nemotron".to_string();
        app.cursor_position = app.search_query.len();
        if !app.selected_providers.is_empty() {
            app.selected_providers.fill(false);
        }
        if !app.selected_use_cases.is_empty() {
            app.selected_use_cases.fill(false);
        }
        if !app.selected_capabilities.is_empty() {
            app.selected_capabilities.fill(false);
        }
        if !app.selected_quants.is_empty() {
            app.selected_quants.fill(false);
        }
        if !app.selected_run_modes.is_empty() {
            app.selected_run_modes.fill(false);
        }
        if !app.selected_params_buckets.is_empty() {
            app.selected_params_buckets.fill(false);
        }

        app.reset_filters();

        assert_eq!(app.fit_filter, FitFilter::All);
        assert_eq!(app.runtime_filter, RuntimeFilter::Any);
        assert_eq!(app.availability_filter, AvailabilityFilter::All);
        assert_eq!(app.tp_filter, TpFilter::All);
        assert_eq!(app.context_filter, ContextFilter::All);
        assert!(!app.installed_first);
        assert_eq!(app.sort_column, llmfit_core::fit::SortColumn::Score);
        assert!(!app.sort_ascending);
        assert!(app.search_query.is_empty());
        assert_eq!(app.cursor_position, 0);
        assert!(app.selected_providers.iter().all(|selected| *selected));
        assert!(app.selected_use_cases.iter().all(|selected| *selected));
        assert!(app.selected_capabilities.iter().all(|selected| *selected));
        assert!(app.selected_quants.iter().all(|selected| *selected));
        assert!(app.selected_run_modes.iter().all(|selected| *selected));
        assert!(app.selected_params_buckets.iter().all(|selected| *selected));
        assert_eq!(app.pull_status.as_deref(), Some("Reset all filters"));
    }

    #[test]
    fn llama_cpp_runtime_filter_rejects_safetensors_without_gguf_path() {
        let (download_capability_tx, download_capability_rx) = mpsc::channel();
        let (catalog_refresh_tx, catalog_refresh_rx) = mpsc::channel();
        let app = App {
            should_quit: false,
            input_mode: super::InputMode::Normal,
            search_query: String::new(),
            cursor_position: 0,
            specs: SystemSpecs {
                total_ram_gb: 64.0,
                available_ram_gb: 48.0,
                total_cpu_cores: 16,
                cpu_name: "Test CPU".to_string(),
                has_gpu: true,
                gpu_vram_gb: Some(24.0),
                total_gpu_vram_gb: Some(24.0),
                gpu_name: Some("RTX 4090".to_string()),
                gpu_count: 1,
                unified_memory: false,
                backend: GpuBackend::Cuda,
                gpus: vec![],
                cluster_mode: false,
                cluster_node_count: 0,
            },
            source_models: vec![],
            base_context_limit: None,
            all_fits: vec![],
            filtered_fits: vec![],
            providers: vec![],
            selected_providers: vec![],
            use_cases: vec![],
            selected_use_cases: vec![],
            capabilities: vec![],
            selected_capabilities: vec![],
            fit_filter: super::FitFilter::All,
            runtime_filter: RuntimeFilter::LlamaCpp,
            availability_filter: super::AvailabilityFilter::All,
            tp_filter: super::TpFilter::All,
            context_filter: super::ContextFilter::All,
            installed_first: false,
            sort_column: llmfit_core::fit::SortColumn::Score,
            sort_ascending: false,
            selected_row: 0,
            show_detail: false,
            show_compare: false,
            compare_mark_model: None,
            show_multi_compare: false,
            compare_models: vec![],
            compare_scroll: 0,
            show_plan: false,
            plan_model_idx: None,
            plan_field: super::PlanField::Context,
            plan_context_input: String::new(),
            plan_quant_input: String::new(),
            plan_target_tps_input: String::new(),
            plan_cursor_position: 0,
            plan_estimate: None,
            plan_error: None,
            provider_cursor: 0,
            use_case_cursor: 0,
            capability_cursor: 0,
            download_provider_cursor: 0,
            download_provider_options: vec![],
            download_provider_model: None,
            ollama_available: false,
            ollama_binary_available: false,
            ollama_installed: HashSet::new(),
            ollama_installed_count: 0,
            ollama: llmfit_core::providers::OllamaProvider::new(),
            mlx_available: false,
            mlx_installed: HashSet::new(),
            mlx: llmfit_core::providers::MlxProvider::new(),
            llamacpp_available: true,
            llamacpp_installed: HashSet::new(),
            llamacpp_installed_count: 0,
            llamacpp_detection_hint: String::new(),
            llamacpp: llmfit_core::providers::LlamaCppProvider::new(),
            docker_mr_available: false,
            docker_mr_installed: HashSet::new(),
            docker_mr_installed_count: 0,
            docker_mr: llmfit_core::providers::DockerModelRunnerProvider::new(),
            lmstudio_available: false,
            lmstudio_installed: HashSet::new(),
            lmstudio_installed_count: 0,
            lmstudio: llmfit_core::providers::LmStudioProvider::new(),
            vllm_available: false,
            vllm_installed: HashSet::new(),
            vllm_installed_count: 0,
            vllm: llmfit_core::providers::VllmProvider::new(),
            pull_active: None,
            pull_status: None,
            pull_percent: None,
            pull_model_name: None,
            pull_provider: None,
            download_capabilities: HashMap::new(),
            download_capability_inflight: HashSet::new(),
            download_capability_tx,
            download_capability_rx,
            catalog_refresh_active: false,
            catalog_refresh_tx,
            catalog_refresh_rx,
            tick_count: 0,
            confirm_download: false,
            visual_anchor: None,
            select_column: 2,
            quants: vec![],
            selected_quants: vec![],
            quant_cursor: 0,
            run_modes: vec![],
            selected_run_modes: vec![],
            run_mode_cursor: 0,
            params_buckets: vec![],
            selected_params_buckets: vec![],
            params_bucket_cursor: 0,
            theme: crate::theme::Theme::Default,
            backend_hidden_count: 0,
        };

        let fit = ModelFit {
            model: LlmModel {
                name: "silx-ai/Quasar-10B".to_string(),
                provider: "huggingface".to_string(),
                parameter_count: "10B".to_string(),
                parameters_raw: Some(10_000_000_000),
                min_ram_gb: 20.0,
                recommended_ram_gb: 32.0,
                min_vram_gb: Some(18.0),
                quantization: "bf16".to_string(),
                context_length: 131_072,
                use_case: "general".to_string(),
                is_moe: false,
                num_experts: None,
                active_experts: None,
                active_parameters: None,
                release_date: None,
                gguf_sources: vec![],
                capabilities: vec![Capability::ToolUse],
                format: ModelFormat::Safetensors,
                num_attention_heads: None,
                num_key_value_heads: None,
                metadata_overlay: None,
            },
            fit_level: FitLevel::Good,
            run_mode: RunMode::Gpu,
            memory_required_gb: 18.0,
            memory_available_gb: 24.0,
            utilization_pct: 75.0,
            notes: vec![],
            moe_offloaded_gb: None,
            score: 90.0,
            score_components: ScoreComponents {
                quality: 90.0,
                speed: 80.0,
                fit: 75.0,
                context: 95.0,
            },
            estimated_tps: 22.0,
            best_quant: "bf16".to_string(),
            use_case: UseCase::Agentic,
            runtime: InferenceRuntime::LlamaCpp,
            installed: false,
        };

        assert!(!app.runtime_filter_matches_fit(&fit));
    }

    #[test]
    fn vllm_runtime_filter_keeps_currently_served_vllm_models() {
        let (download_capability_tx, download_capability_rx) = mpsc::channel();
        let (catalog_refresh_tx, catalog_refresh_rx) = mpsc::channel();
        let mut vllm_installed = HashSet::new();
        vllm_installed.insert("silx-ai/quasar-10b".to_string());
        let app = App {
            should_quit: false,
            input_mode: super::InputMode::Normal,
            search_query: String::new(),
            cursor_position: 0,
            specs: SystemSpecs {
                total_ram_gb: 64.0,
                available_ram_gb: 48.0,
                total_cpu_cores: 16,
                cpu_name: "Test CPU".to_string(),
                has_gpu: true,
                gpu_vram_gb: Some(24.0),
                total_gpu_vram_gb: Some(24.0),
                gpu_name: Some("RTX 4090".to_string()),
                gpu_count: 1,
                unified_memory: false,
                backend: GpuBackend::Cuda,
                gpus: vec![],
                cluster_mode: false,
                cluster_node_count: 0,
            },
            source_models: vec![],
            base_context_limit: None,
            all_fits: vec![],
            filtered_fits: vec![],
            providers: vec![],
            selected_providers: vec![],
            use_cases: vec![],
            selected_use_cases: vec![],
            capabilities: vec![],
            selected_capabilities: vec![],
            fit_filter: super::FitFilter::All,
            runtime_filter: RuntimeFilter::Vllm,
            availability_filter: super::AvailabilityFilter::All,
            tp_filter: super::TpFilter::All,
            context_filter: super::ContextFilter::All,
            installed_first: false,
            sort_column: llmfit_core::fit::SortColumn::Score,
            sort_ascending: false,
            selected_row: 0,
            show_detail: false,
            show_compare: false,
            compare_mark_model: None,
            show_multi_compare: false,
            compare_models: vec![],
            compare_scroll: 0,
            show_plan: false,
            plan_model_idx: None,
            plan_field: super::PlanField::Context,
            plan_context_input: String::new(),
            plan_quant_input: String::new(),
            plan_target_tps_input: String::new(),
            plan_cursor_position: 0,
            plan_estimate: None,
            plan_error: None,
            provider_cursor: 0,
            use_case_cursor: 0,
            capability_cursor: 0,
            download_provider_cursor: 0,
            download_provider_options: vec![],
            download_provider_model: None,
            ollama_available: false,
            ollama_binary_available: false,
            ollama_installed: HashSet::new(),
            ollama_installed_count: 0,
            ollama: llmfit_core::providers::OllamaProvider::new(),
            mlx_available: false,
            mlx_installed: HashSet::new(),
            mlx: llmfit_core::providers::MlxProvider::new(),
            llamacpp_available: false,
            llamacpp_installed: HashSet::new(),
            llamacpp_installed_count: 0,
            llamacpp_detection_hint: String::new(),
            llamacpp: llmfit_core::providers::LlamaCppProvider::new(),
            docker_mr_available: false,
            docker_mr_installed: HashSet::new(),
            docker_mr_installed_count: 0,
            docker_mr: llmfit_core::providers::DockerModelRunnerProvider::new(),
            lmstudio_available: false,
            lmstudio_installed: HashSet::new(),
            lmstudio_installed_count: 0,
            lmstudio: llmfit_core::providers::LmStudioProvider::new(),
            vllm_available: true,
            vllm_installed,
            vllm_installed_count: 1,
            vllm: llmfit_core::providers::VllmProvider::new(),
            pull_active: None,
            pull_status: None,
            pull_percent: None,
            pull_model_name: None,
            pull_provider: None,
            download_capabilities: HashMap::new(),
            download_capability_inflight: HashSet::new(),
            download_capability_tx,
            download_capability_rx,
            catalog_refresh_active: false,
            catalog_refresh_tx,
            catalog_refresh_rx,
            tick_count: 0,
            confirm_download: false,
            visual_anchor: None,
            select_column: 2,
            quants: vec![],
            selected_quants: vec![],
            quant_cursor: 0,
            run_modes: vec![],
            selected_run_modes: vec![],
            run_mode_cursor: 0,
            params_buckets: vec![],
            selected_params_buckets: vec![],
            params_bucket_cursor: 0,
            theme: crate::theme::Theme::Default,
            backend_hidden_count: 0,
        };

        let fit = ModelFit {
            model: LlmModel {
                name: "silx-ai/Quasar-10B".to_string(),
                provider: "huggingface".to_string(),
                parameter_count: "10B".to_string(),
                parameters_raw: Some(10_000_000_000),
                min_ram_gb: 20.0,
                recommended_ram_gb: 32.0,
                min_vram_gb: Some(18.0),
                quantization: "bf16".to_string(),
                context_length: 131_072,
                use_case: "general".to_string(),
                is_moe: false,
                num_experts: None,
                active_experts: None,
                active_parameters: None,
                release_date: None,
                gguf_sources: vec![],
                capabilities: vec![Capability::ToolUse],
                format: ModelFormat::Safetensors,
                num_attention_heads: None,
                num_key_value_heads: None,
                metadata_overlay: None,
            },
            fit_level: FitLevel::Good,
            run_mode: RunMode::Gpu,
            memory_required_gb: 18.0,
            memory_available_gb: 24.0,
            utilization_pct: 75.0,
            notes: vec![],
            moe_offloaded_gb: None,
            score: 90.0,
            score_components: ScoreComponents {
                quality: 90.0,
                speed: 80.0,
                fit: 75.0,
                context: 95.0,
            },
            estimated_tps: 22.0,
            best_quant: "bf16".to_string(),
            use_case: UseCase::Agentic,
            runtime: InferenceRuntime::LlamaCpp,
            installed: true,
        };

        assert!(app.runtime_filter_matches_fit(&fit));
    }

    #[test]
    fn llama_cpp_runtime_filter_keeps_actual_gguf_models() {
        let (download_capability_tx, download_capability_rx) = mpsc::channel();
        let (catalog_refresh_tx, catalog_refresh_rx) = mpsc::channel();
        let app = App {
            should_quit: false,
            input_mode: super::InputMode::Normal,
            search_query: String::new(),
            cursor_position: 0,
            specs: SystemSpecs {
                total_ram_gb: 64.0,
                available_ram_gb: 48.0,
                total_cpu_cores: 16,
                cpu_name: "Test CPU".to_string(),
                has_gpu: true,
                gpu_vram_gb: Some(24.0),
                total_gpu_vram_gb: Some(24.0),
                gpu_name: Some("RTX 4090".to_string()),
                gpu_count: 1,
                unified_memory: false,
                backend: GpuBackend::Cuda,
                gpus: vec![],
                cluster_mode: false,
                cluster_node_count: 0,
            },
            source_models: vec![],
            base_context_limit: None,
            all_fits: vec![],
            filtered_fits: vec![],
            providers: vec![],
            selected_providers: vec![],
            use_cases: vec![],
            selected_use_cases: vec![],
            capabilities: vec![],
            selected_capabilities: vec![],
            fit_filter: super::FitFilter::All,
            runtime_filter: RuntimeFilter::LlamaCpp,
            availability_filter: super::AvailabilityFilter::All,
            tp_filter: super::TpFilter::All,
            context_filter: super::ContextFilter::All,
            installed_first: false,
            sort_column: llmfit_core::fit::SortColumn::Score,
            sort_ascending: false,
            selected_row: 0,
            show_detail: false,
            show_compare: false,
            compare_mark_model: None,
            show_multi_compare: false,
            compare_models: vec![],
            compare_scroll: 0,
            show_plan: false,
            plan_model_idx: None,
            plan_field: super::PlanField::Context,
            plan_context_input: String::new(),
            plan_quant_input: String::new(),
            plan_target_tps_input: String::new(),
            plan_cursor_position: 0,
            plan_estimate: None,
            plan_error: None,
            provider_cursor: 0,
            use_case_cursor: 0,
            capability_cursor: 0,
            download_provider_cursor: 0,
            download_provider_options: vec![],
            download_provider_model: None,
            ollama_available: false,
            ollama_binary_available: false,
            ollama_installed: HashSet::new(),
            ollama_installed_count: 0,
            ollama: llmfit_core::providers::OllamaProvider::new(),
            mlx_available: false,
            mlx_installed: HashSet::new(),
            mlx: llmfit_core::providers::MlxProvider::new(),
            llamacpp_available: true,
            llamacpp_installed: HashSet::new(),
            llamacpp_installed_count: 0,
            llamacpp_detection_hint: String::new(),
            llamacpp: llmfit_core::providers::LlamaCppProvider::new(),
            docker_mr_available: false,
            docker_mr_installed: HashSet::new(),
            docker_mr_installed_count: 0,
            docker_mr: llmfit_core::providers::DockerModelRunnerProvider::new(),
            lmstudio_available: false,
            lmstudio_installed: HashSet::new(),
            lmstudio_installed_count: 0,
            lmstudio: llmfit_core::providers::LmStudioProvider::new(),
            vllm_available: false,
            vllm_installed: HashSet::new(),
            vllm_installed_count: 0,
            vllm: llmfit_core::providers::VllmProvider::new(),
            pull_active: None,
            pull_status: None,
            pull_percent: None,
            pull_model_name: None,
            pull_provider: None,
            download_capabilities: HashMap::new(),
            download_capability_inflight: HashSet::new(),
            download_capability_tx,
            download_capability_rx,
            catalog_refresh_active: false,
            catalog_refresh_tx,
            catalog_refresh_rx,
            tick_count: 0,
            confirm_download: false,
            visual_anchor: None,
            select_column: 2,
            quants: vec![],
            selected_quants: vec![],
            quant_cursor: 0,
            run_modes: vec![],
            selected_run_modes: vec![],
            run_mode_cursor: 0,
            params_buckets: vec![],
            selected_params_buckets: vec![],
            params_bucket_cursor: 0,
            theme: crate::theme::Theme::Default,
            backend_hidden_count: 0,
        };

        let fit = ModelFit {
            model: LlmModel {
                name: "Qwen/Qwen2.5-7B-Instruct".to_string(),
                provider: "huggingface".to_string(),
                parameter_count: "7B".to_string(),
                parameters_raw: Some(7_000_000_000),
                min_ram_gb: 8.0,
                recommended_ram_gb: 12.0,
                min_vram_gb: Some(4.0),
                quantization: "Q4_K_M".to_string(),
                context_length: 131_072,
                use_case: "chat".to_string(),
                is_moe: false,
                num_experts: None,
                active_experts: None,
                active_parameters: None,
                release_date: None,
                gguf_sources: vec![llmfit_core::models::GgufSource {
                    repo: "bartowski/Qwen2.5-7B-Instruct-GGUF".to_string(),
                    provider: "bartowski".to_string(),
                }],
                capabilities: vec![],
                format: ModelFormat::Gguf,
                num_attention_heads: None,
                num_key_value_heads: None,
                metadata_overlay: None,
            },
            fit_level: FitLevel::Good,
            run_mode: RunMode::Gpu,
            memory_required_gb: 4.0,
            memory_available_gb: 24.0,
            utilization_pct: 16.0,
            notes: vec![],
            moe_offloaded_gb: None,
            score: 92.0,
            score_components: ScoreComponents {
                quality: 88.0,
                speed: 91.0,
                fit: 96.0,
                context: 95.0,
            },
            estimated_tps: 45.0,
            best_quant: "Q4_K_M".to_string(),
            use_case: UseCase::Chat,
            runtime: InferenceRuntime::LlamaCpp,
            installed: false,
        };

        assert!(app.runtime_filter_matches_fit(&fit));
    }

    #[test]
    fn lmstudio_download_prefers_catalog_lmstudio_repo() {
        let mut app = App::with_specs_and_context(
            SystemSpecs {
                total_ram_gb: 64.0,
                available_ram_gb: 48.0,
                total_cpu_cores: 16,
                cpu_name: "Test CPU".to_string(),
                has_gpu: true,
                gpu_vram_gb: Some(24.0),
                total_gpu_vram_gb: Some(24.0),
                gpu_name: Some("RTX 4090".to_string()),
                gpu_count: 1,
                unified_memory: false,
                backend: GpuBackend::Cuda,
                gpus: vec![],
                cluster_mode: false,
                cluster_node_count: 0,
            },
            None,
        );

        app.all_fits = vec![ModelFit {
            model: LlmModel {
                name: "HuggingFaceTB/SmolLM3-3B".to_string(),
                provider: "huggingface".to_string(),
                parameter_count: "3B".to_string(),
                parameters_raw: Some(3_000_000_000),
                min_ram_gb: 2.0,
                recommended_ram_gb: 4.0,
                min_vram_gb: Some(2.0),
                quantization: "Q4_K_M".to_string(),
                context_length: 65_536,
                use_case: "general".to_string(),
                is_moe: false,
                num_experts: None,
                active_experts: None,
                active_parameters: None,
                release_date: None,
                gguf_sources: vec![
                    llmfit_core::models::GgufSource {
                        repo: "unsloth/SmolLM3-3B-GGUF".to_string(),
                        provider: "unsloth".to_string(),
                    },
                    llmfit_core::models::GgufSource {
                        repo: "lmstudio-community/SmolLM3-3B-GGUF".to_string(),
                        provider: "lmstudio-community".to_string(),
                    },
                ],
                capabilities: vec![],
                format: ModelFormat::Gguf,
                num_attention_heads: None,
                num_key_value_heads: None,
                metadata_overlay: None,
            },
            fit_level: FitLevel::Good,
            run_mode: RunMode::Gpu,
            memory_required_gb: 2.0,
            memory_available_gb: 24.0,
            utilization_pct: 10.0,
            notes: vec![],
            moe_offloaded_gb: None,
            score: 90.0,
            score_components: ScoreComponents {
                quality: 90.0,
                speed: 80.0,
                fit: 95.0,
                context: 85.0,
            },
            estimated_tps: 42.0,
            best_quant: "Q4_K_M".to_string(),
            use_case: UseCase::General,
            runtime: InferenceRuntime::LlamaCpp,
            installed: false,
        }];

        assert_eq!(
            app.lmstudio_download_candidates("HuggingFaceTB/SmolLM3-3B"),
            vec![
                "lmstudio-community/SmolLM3-3B-GGUF".to_string(),
                "https://huggingface.co/lmstudio-community/SmolLM3-3B-GGUF".to_string(),
                "SmolLM3-3B-GGUF".to_string(),
                "huggingfacetb/smollm3-3b".to_string(),
                "smollm3-3b".to_string(),
                "unsloth/SmolLM3-3B-GGUF".to_string(),
                "https://huggingface.co/unsloth/SmolLM3-3B-GGUF".to_string(),
            ]
        );
    }
}
