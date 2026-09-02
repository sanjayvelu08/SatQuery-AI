import { MapPin, Layers, Clock, Zap } from 'lucide-react';
import type { ChangeResult } from '../types';

interface ChangeResultPanelProps {
  change: ChangeResult;
  originalImageUrl: string | null;
}

export function ChangeResultPanel({ change, originalImageUrl }: ChangeResultPanelProps) {
  const changePct = Math.round(change.change_pct * 10) / 10;

  return (
    <div className="space-y-4 animate-fade-in-up">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg">🔄</span>
          <h3 className="text-sm font-semibold text-text-primary">Change Detection</h3>
        </div>
        <span className="text-xs font-mono text-text-muted">{change.model_used}</span>
      </div>

      {/* Change percentage bar */}
      <div className="bg-bg-card border border-border rounded-lg p-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-text-secondary">Changed Area</span>
          <span className="text-sm font-bold text-accent-teal">{changePct}%</span>
        </div>
        <div className="h-2 rounded-full bg-white/10 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-accent-teal to-accent-coral transition-all duration-700"
            style={{ width: `${Math.min(changePct, 100)}%` }}
          />
        </div>
        <p className="text-[11px] text-text-muted mt-2">{change.summary}</p>
      </div>

      {/* Visualizations */}
      <div className="grid grid-cols-2 gap-2">
        {change.overlay_url && (
          <div className="rounded-lg overflow-hidden border border-border">
            <img
              src={change.overlay_url}
              alt="Change overlay"
              className="w-full h-auto"
            />
            <div className="px-2 py-1 bg-bg-card text-[10px] text-text-muted text-center">
              Change Overlay
            </div>
          </div>
        )}
        {change.bbox_url && (
          <div className="rounded-lg overflow-hidden border border-border">
            <img
              src={change.bbox_url}
              alt="Change bounding boxes"
              className="w-full h-auto"
            />
            <div className="px-2 py-1 bg-bg-card text-[10px] text-text-muted text-center">
              Bounding Boxes
            </div>
          </div>
        )}
        {change.mask_url && (
          <div className="rounded-lg overflow-hidden border border-border">
            <img
              src={change.mask_url}
              alt="Change mask"
              className="w-full h-auto"
            />
            <div className="px-2 py-1 bg-bg-card text-[10px] text-text-muted text-center">
              Binary Mask
            </div>
          </div>
        )}
        {originalImageUrl && !change.overlay_url && (
          <div className="rounded-lg overflow-hidden border border-border">
            <img
              src={originalImageUrl}
              alt="Original"
              className="w-full h-auto opacity-80"
            />
            <div className="px-2 py-1 bg-bg-card text-[10px] text-text-muted text-center">
              Original
            </div>
          </div>
        )}
      </div>

      {/* Changed regions */}
      {change.regions.length > 0 && (
        <div className="bg-bg-card border border-border rounded-lg overflow-hidden">
          <div className="px-3 py-2 border-b border-border bg-accent-teal/5">
            <div className="flex items-center gap-2">
              <MapPin className="w-3 h-3 text-accent-teal" />
              <span className="text-xs font-semibold text-text-secondary">
                {change.regions.length} Changed Region{change.regions.length !== 1 ? 's' : ''}
              </span>
            </div>
          </div>
          <div className="divide-y divide-border/50">
            {change.regions.map((region) => (
              <div key={region.id} className="px-3 py-2 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono bg-bg-primary px-1.5 py-0.5 rounded text-text-muted">
                    #{region.id}
                  </span>
                  <span className="text-xs text-text-secondary">
                    {region.area_pct.toFixed(1)}% of image
                  </span>
                </div>
                <span className="text-[10px] font-mono text-text-muted">
                  [{region.bbox.join(', ')}]
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Performance */}
      <div className="flex items-center gap-4 text-[11px] text-text-muted">
        <span className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {(change.total_ms / 1000).toFixed(1)}s
        </span>
        <span className="flex items-center gap-1">
          <Zap className="w-3 h-3" />
          {change.vram_peak_mb.toFixed(0)} MB VRAM
        </span>
        <span className="flex items-center gap-1">
          <Layers className="w-3 h-3" />
          {change.img_size}×{change.img_size}
        </span>
      </div>
    </div>
  );
}
