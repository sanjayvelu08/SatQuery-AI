import type { SARDetection } from '../types';
import { ConfidenceBar } from './ConfidenceBar';

interface DetectionTableProps {
  detections: SARDetection[];
  type: 'sar' | 'grounding';
}

export function DetectionTable({ detections, type }: DetectionTableProps) {
  if (detections.length === 0) return null;

  return (
    <div className="rounded-lg overflow-hidden border border-border animate-fade-in-up">
      <table className="w-full text-left">
        <thead>
          <tr className="bg-accent-teal/8">
            <th className="px-3 py-1.5 text-[10px] font-semibold text-text-secondary uppercase tracking-wider border-b border-border">
              #
            </th>
            <th className="px-3 py-1.5 text-[10px] font-semibold text-text-secondary uppercase tracking-wider border-b border-border">
              Object
            </th>
            <th className="px-3 py-1.5 text-[10px] font-semibold text-text-secondary uppercase tracking-wider border-b border-border">
              Confidence
            </th>
            {type === 'sar' && (
              <th className="px-3 py-1.5 text-[10px] font-semibold text-text-secondary uppercase tracking-wider border-b border-border hidden sm:table-cell">
                Bounding Box
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {detections.map((det, i) => (
            <tr key={i} className="hover:bg-accent-teal/4 transition-colors border-b border-border/40 last:border-0">
              <td className="px-3 py-2 text-[11px] font-mono text-text-muted">
                {i + 1}
              </td>
              <td className="px-3 py-2 text-sm font-medium text-text-primary capitalize">
                {det.class_name}
              </td>
              <td className="px-3 py-2">
                <ConfidenceBar confidence={det.confidence} />
              </td>
              {type === 'sar' && (
                <td className="px-3 py-2 text-[11px] font-mono text-text-muted hidden sm:table-cell">
                  [{det.bbox_xyxy.map((v) => Math.round(v)).join(', ')}]
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
