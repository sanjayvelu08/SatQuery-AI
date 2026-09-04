import type { GroundingDetection } from '../types';

interface GroundingPanelProps {
  target: string;
  detections: GroundingDetection[];
}

export function GroundingPanel({ target, detections }: GroundingPanelProps) {
  return (
    <div className="rounded-lg overflow-hidden border border-border animate-fade-in-up">
      <div className="px-3 py-2 bg-accent-teal/8 border-b border-border flex items-center justify-between">
        <span className="text-[10px] font-semibold text-text-secondary uppercase tracking-wider">
          📍 Text-Guided Grounding
        </span>
        <span className="text-[11px] font-medium text-text-secondary truncate max-w-[60%]">
          target: <span className="text-accent-teal font-semibold">{target}</span>
        </span>
        <span className="text-[10px] font-mono text-text-muted shrink-0">
          {detections.length} box{detections.length === 1 ? '' : 'es'}
        </span>
      </div>
      {detections.length === 0 ? (
        <div className="px-3 py-4 text-xs text-text-secondary">
          No bounding boxes matched the target in this image.
        </div>
      ) : (
        <table className="w-full text-left">
          <thead>
            <tr className="bg-accent-teal/8">
              <th className="px-3 py-1.5 text-[10px] font-semibold text-text-secondary uppercase tracking-wider border-b border-border">
                #
              </th>
              <th className="px-3 py-1.5 text-[10px] font-semibold text-text-secondary uppercase tracking-wider border-b border-border">
                Bounding Box (px)
              </th>
              <th className="px-3 py-1.5 text-[10px] font-semibold text-text-secondary uppercase tracking-wider border-b border-border">
                Normalized (0-100)
              </th>
              <th className="px-3 py-1.5 text-[10px] font-semibold text-text-secondary uppercase tracking-wider border-b border-border">
                Confidence
              </th>
            </tr>
          </thead>
          <tbody>
            {detections.map((d, i) => (
              <tr key={i} className="hover:bg-accent-teal/4 transition-colors border-b border-border/40 last:border-0">
                <td className="px-3 py-2 text-[11px] font-mono text-text-muted">{i + 1}</td>
                <td className="px-3 py-2 text-[11px] font-mono text-text-primary">
                  [{Math.round(d.x1)}, {Math.round(d.y1)}, {Math.round(d.x2)}, {Math.round(d.y2)}]
                </td>
                <td className="px-3 py-2 text-[11px] font-mono text-text-muted hidden sm:table-cell">
                  [{d.x1_norm.toFixed(1)}, {d.y1_norm.toFixed(1)}, {d.x2_norm.toFixed(1)}, {d.y2_norm.toFixed(1)}]
                </td>
                <td className="px-3 py-2 text-[11px] font-mono text-text-muted">
                  {d.confidence != null ? `${(d.confidence * 100).toFixed(0)}%` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}