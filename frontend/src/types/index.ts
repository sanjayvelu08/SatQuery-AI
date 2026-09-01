export type IntentType =
  | 'caption'
  | 'vqa'
  | 'detect'
  | 'grounding'
  | 'classification'
  | 'change'
  | 'sar'
  | 'general';

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

export interface AnalyzeResponse {
  query: string;
  intent: IntentType;
  all_intents: IntentType[];
  supported: boolean;
  answer: string | null;
  unsupported_reason: string;
  model_used: string;
  annotated_image_url: string | null;
  elapsed_route_ms: number;
  elapsed_vlm_s: number;
  elapsed_total_s: number;
  sar_result: SARResult | null;
}

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
}
