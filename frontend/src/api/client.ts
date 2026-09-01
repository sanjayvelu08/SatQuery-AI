import type { AnalyzeResponse, DemoScenario } from '../types';

const API_BASE = '';

export async function analyzeImage(
  image: File,
  query: string,
): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append('image', image);
  formData.append('query', query);

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Analysis failed: ${res.status}`);
  }

  return res.json();
}

export async function analyzeDemo(
  demoName: string,
): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append('query', demoName);
  formData.append('demo', demoName);

  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Demo analysis failed: ${res.status}`);
  }

  return res.json();
}

export async function fetchDemos(): Promise<DemoScenario[]> {
  const res = await fetch(`${API_BASE}/api/demos`);
  if (!res.ok) throw new Error(`Failed to fetch demos: ${res.status}`);
  return res.json();
}

export async function checkHealth(): Promise<{ status: string; vram: Record<string, number> }> {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export function getAnnotatedUrl(path: string | null): string | null {
  if (!path) return null;
  // Backend returns relative path like "annotated/xxx.jpg"
  if (path.startsWith('http')) return path;
  return `${API_BASE}/${path}`;
}
