import { Check, X, SkipForward, Clock } from 'lucide-react';
import type { ExecutionTraceStep } from '../types';

interface TraceTimelineProps {
  steps: ExecutionTraceStep[];
  totalMs: number;
}

const STEP_ICONS: Record<string, string> = {
  route: '🔍',
  validate: '✅',
  geo_probe: '🗺️',
  pair_compat_check: '🧭',
  specialist_selection: '🎯',
  sar_detect: '📡',
  optical_analyze: '👁️',
  fuse: '🔗',
  interpret: '💬',
  confidence: '📊',
};

export function TraceTimeline({ steps, totalMs }: TraceTimelineProps) {
  return (
    <div className="bg-bg-card border border-border rounded-lg overflow-hidden">
      <div className="px-3 py-2 border-b border-border flex items-center justify-between">
        <span className="text-[11px] font-semibold text-text-secondary">Execution Trace</span>
        <span className="text-[10px] font-mono text-text-muted flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {(totalMs / 1000).toFixed(1)}s total
        </span>
      </div>
      <div className="divide-y divide-border/40">
        {steps.map((step, i) => (
          <div key={i} className="px-3 py-2">
            <div className="flex items-center gap-3">
              {/* Status icon */}
              <div className="shrink-0">
                {step.status === 'ok' ? (
                  <Check className="w-3.5 h-3.5 text-accent-green" />
                ) : step.status === 'error' ? (
                  <X className="w-3.5 h-3.5 text-accent-coral" />
                ) : (
                  <SkipForward className="w-3.5 h-3.5 text-text-muted" />
                )}
              </div>

              {/* Step icon + name */}
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-xs">{STEP_ICONS[step.name] || '📌'}</span>
                <span className="text-[11px] font-medium text-text-primary truncate">
                  {step.name.replace(/_/g, ' ')}
                </span>
              </div>

              {/* Tool */}
              <span className="text-[9px] font-mono text-text-muted hidden sm:block shrink-0">
                {step.tool}
              </span>

              {/* Output */}
              <span className="text-[10px] text-text-muted truncate flex-1 min-w-0">
                {step.output_summary}
              </span>

              {/* Duration */}
              <span className="text-[10px] font-mono text-text-muted shrink-0">
                {step.duration_ms >= 1000
                  ? `${(step.duration_ms / 1000).toFixed(1)}s`
                  : `${step.duration_ms.toFixed(0)}ms`}
              </span>
            </div>

            {/* Audit details: inputs used + safe failure reason */}
            {(step.input_summary || step.error) && (
              <div className="mt-1 pl-8 flex flex-col gap-0.5 min-w-0">
                {step.input_summary && step.input_summary !== step.output_summary && (
                  <span className="text-[9.5px] font-mono text-text-muted/80 truncate">
                    in: {step.input_summary}
                  </span>
                )}
                {step.error && (
                  <span className="text-[9.5px] font-mono text-accent-coral truncate">
                    error: {step.error}
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
