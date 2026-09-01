import { clsx, type ClassValue } from 'clsx';

export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs);
}

export function formatTiming(routeMs: number, vlmS: number, totalS: number): string {
  const parts: string[] = [];
  if (routeMs > 0) parts.push(`Route ${Math.round(routeMs)}ms`);
  if (vlmS > 0) parts.push(`VLM ${vlmS.toFixed(1)}s`);
  if (totalS > 0) parts.push(`Total ${totalS.toFixed(1)}s`);
  if (parts.length === 0) return '⚡ Instant';
  return parts.join(' · ');
}

export function truncate(str: string, max: number): string {
  if (str.length <= max) return str;
  return str.slice(0, max - 1) + '…';
}
