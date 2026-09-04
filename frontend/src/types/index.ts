export type IntentType =
  | 'caption'
  | 'vqa'
  | 'detect'
  | 'grounding'
  | 'classification'
  | 'change'
  | 'sar'
  | 'joint_analysis'
  | 'general';

// ── SAR ──────────────────────────────────────────────────────

export interface SARDetection {
  class_name: string;
  confidence: number;
  bbox_xyxy: number[];
}

export interface SARResult {
  success: boolean;
  detections: SARDetection[];
  num_detections: number;
  inference_time_ms: number;
  gpu_vram_mb: number;
  error: string | null;
}

// ── Change Detection ─────────────────────────────────────────

export interface ChangeRegion {
  id: number;
  bbox: number[];
  area_px: number;
  area_pct: number;
  w: number;
  h: number;
}

export interface ChangeResult {
  success: boolean;
  change_detected: boolean;
  change_pct: number;
  num_regions: number;
  regions: ChangeRegion[];
  summary: string;
  model_used: string;
  inference_time_ms: number;
  total_ms: number;
  vram_peak_mb: number;
  img_size: number;
  overlay_url: string | null;
  bbox_url: string | null;
  mask_url: string | null;
}

// ── Joint Analysis ───────────────────────────────────────────

export interface OpticalEvidence {
  success: boolean;
  elapsed_s: number;
  error: string | null;
}

export interface SAREvidence {
  success: boolean;
  num_detections: number;
  detection_summary: string;
  inference_time_ms: number;
  error: string | null;
}

export interface ExecutionTraceStep {
  step: number;
  name: string;
  tool: string;
  status: string;
  duration_ms: number;
  input_summary: string;
  output_summary: string;
  error?: string;
}

export interface JointResult {
  query: string;
  joint_answer: string;
  confidence: number;
  confidence_reasoning: string;
  total_ms: number;
  models_used: string[];
  trace: ExecutionTraceStep[];
  sar?: SAREvidence;
  optical?: OpticalEvidence;
  composite_url?: string | null;
}

// ── Main Response ────────────────────────────────────────────

export interface AnalysisSummary {
  query: string;
  intent: IntentType;
  models_used: string;
  evidence_reliability: number | null;
  reliability_reasoning: string | null;
  reliability_note: string | null;
  warnings: string[];
  trace_step_count: number;
}

export interface AnalyzeResponse {
  query: string;
  intent: IntentType;
  all_intents: IntentType[];
  supported: boolean;
  answer: string | null;
  unsupported_reason: string;
  model_used: string;
  annotated_image_url: string | null;
  change_result: ChangeResult | null;
  joint_result: JointResult | null;
  trace: ExecutionTraceStep[];
  elapsed_route_ms: number;
  elapsed_vlm_s: number;
  elapsed_total_s: number;
  sar_result: SARResult | null;
  summary?: AnalysisSummary;
}

// ── Demo / History ───────────────────────────────────────────

export interface DemoScenario {
  name: string;
  image_url: string;
  query: string;
  intent: IntentType;
  model_used: string;
  answer: string;
  all_intents: IntentType[];
  supported: boolean;
}

export type AnalysisMode = 'single' | 'change' | 'joint';

export interface HistoryEntry {
  id: string;
  timestamp: number;
  query: string;
  intent: IntentType;
  model_used: string;
  elapsed_s: number;
  supported: boolean;
  result: AnalyzeResponse;
  imagePreview: string;
  mode: AnalysisMode;
}
