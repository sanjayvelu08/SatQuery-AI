import type { IntentType } from '../types';

export const INTENT_COLORS: Record<IntentType, string> = {
  caption: '#4FC3F7',
  vqa: '#AB47BC',
  detect: '#EF5350',
  grounding: '#FF7043',
  classification: '#FFA726',
  change: '#78909C',
  sar: '#26A69A',
  joint_analysis: '#7C4DFF',
  general: '#78909C',
};

export const INTENT_LABELS: Record<IntentType, string> = {
  caption: 'Captioning',
  vqa: 'Visual QA',
  detect: 'Detection',
  grounding: 'Grounding',
  classification: 'Classification',
  change: 'Change Detection',
  sar: 'SAR Analysis',
  joint_analysis: 'Joint Analysis',
  general: 'General',
};

export const INTENT_ICONS: Record<IntentType, string> = {
  caption: '📝',
  vqa: '❓',
  detect: '🎯',
  grounding: '📍',
  classification: '🏷️',
  change: '🔄',
  sar: '📡',
  joint_analysis: '🔗',
  general: '💬',
};

export const MODEL_LABELS: Record<string, string> = {
  'EarthDial 4B RGB': 'EarthDial 4B RGB (VLM)',
  'EarthDial 4B RGB (VLM)': 'EarthDial 4B RGB (VLM)',
  'EarthDial 4B RGB (VLM + Grounding)': 'EarthDial 4B RGB (VLM + Grounding)',
  'YOLOv8 SAR Vessel Detector': 'YOLOv8 SAR Vessel Detector',
  'YOLOv8 SAR': 'YOLOv8 SAR Vessel Detector',
  'EarthDial 4B + YOLOv8 SAR (Joint)': 'EarthDial + SAR (Joint)',
  'BIT-CD (LEVIR-CD pretrained)': 'BIT-CD Change Detection',
};

export const CAPABILITY_CARDS = [
  { icon: '📝', title: 'Captioning', desc: 'Describe satellite imagery in detail', intent: 'caption' as IntentType },
  { icon: '❓', title: 'Visual QA', desc: 'Ask questions about what you see', intent: 'vqa' as IntentType },
  { icon: '🎯', title: 'Detection', desc: 'Locate features with bounding boxes', intent: 'detect' as IntentType },
  { icon: '📡', title: 'SAR Ships', desc: 'Detect vessels in radar imagery', intent: 'sar' as IntentType },
];
