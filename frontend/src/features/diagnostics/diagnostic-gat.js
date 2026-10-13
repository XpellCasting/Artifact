export const DEFAULT_GAT_CONFIG = Object.freeze({
  method: "gat_lstm", schema_version: "gat-lstm-segments-v1", task: "diagnostic_3class",
  architecture_version: "gat_lstm_v3_segments_multiscale",
  segment_threshold: .5, local_timezone: "America/Santiago", entity_value_modes: null,
  segment_encoder: "gat", active_scales: Object.freeze([true, true, true]),
  grid_seconds: 60, sequence_steps: 60, feature_window_seconds: 900,
  labelled_stride_seconds: 300, graph_top_k: 4, max_nodes: 32, entropy_bins: 16,
  dtw_reference_count: 3, dtw_band_fraction: .1, gat_heads: 4, gat_head_dim: 8,
  lstm_hidden: 32, dropout: .1, batch_size: 16, learning_rate: .001,
  max_epochs: 30, patience: 5, seed: 42, device: "cpu", pretrain_enabled: false,
  pretrain_stride_seconds: 900, pretrain_max_samples: 4096, use_entropy: true, use_dtw: true,
});

export const GAT_LABELS = ["normal", "environmental_anomaly", "sensor_fault"];
export const SEGMENT_ARCHITECTURE = "gat_lstm_v3_segments_multiscale";
export const DEFAULT_SEGMENT_CONFIG = DEFAULT_GAT_CONFIG;
export const SEGMENT_SCALES = [
  { label: "Corta", hours: 1, resolutionMinutes: 1 },
  { label: "Diaria", hours: 24, resolutionMinutes: 15 },
  { label: "Extendida", hours: 72, resolutionMinutes: 60 },
];
export const GAT_TRAIN_PARTITIONS = ["train_fit", "dev_tune"];

export function isGatConfigValid(config) {
  return config.architecture_version === SEGMENT_ARCHITECTURE && !config.pretrain_enabled
    && (config.schema_version === "gat-lstm-segments-v1"
      && Number.isFinite(config.segment_threshold) && config.segment_threshold > 0 && config.segment_threshold < 1
      && ["gat", "lstm_only"].includes(config.segment_encoder || "gat")
      && (config.active_scales || [true,true,true]).length === 3
      && (config.active_scales || [true,true,true]).every(active => typeof active === "boolean")
      && (config.active_scales || [true,true,true]).some(Boolean))
    && Number.isInteger(config.gat_heads) && Number.isInteger(config.gat_head_dim)
    && config.gat_heads * config.gat_head_dim === 32
    && Number.isInteger(config.sequence_steps) && config.sequence_steps >= 2
    && Number.isInteger(config.grid_seconds) && config.grid_seconds > 0
    && Number.isInteger(config.feature_window_seconds) && config.feature_window_seconds > 0;
}

export function gatSupportStatus(dataset) {
  const support = dataset?.label_counts || {};
  const missing = GAT_TRAIN_PARTITIONS.flatMap((partition) => GAT_LABELS
    .filter((label) => !Number(support[partition]?.[label] || 0)
      || !Number(dataset?.episode_counts?.[partition]?.[label] || 0))
    .map((label) => ({ partition, label })));
  return { ready: Boolean(dataset) && missing.length === 0, missing };
}

export function gatDataConfigCompatible(config, dataset) {
  if (!dataset?.config || !isGatConfigValid(config)) return false;
  const fields = ["grid_seconds", "sequence_steps", "feature_window_seconds", "graph_top_k",
    "max_nodes", "use_entropy", "use_dtw", "architecture_version"];
  if (config.architecture_version === SEGMENT_ARCHITECTURE &&
      ((config.local_timezone || "America/Santiago") !== (dataset.config.local_timezone || "America/Santiago")
       || JSON.stringify(config.entity_value_modes || {}) !== JSON.stringify(dataset.config.entity_value_modes || {})
       || JSON.stringify(config.active_scales || [true,true,true]) !== JSON.stringify(dataset.config.active_scales || [true,true,true]))) return false;
  return fields.every((key) => key === "architecture_version"
    ? config[key] === (dataset.config[key])
    : config[key] === dataset.config[key]);
}

export function gatContextSeconds(config) {
  return config.architecture_version === SEGMENT_ARCHITECTURE ? 72 * 3600 : null;
}
