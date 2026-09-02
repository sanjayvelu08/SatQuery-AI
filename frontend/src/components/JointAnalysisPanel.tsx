import { Shield, AlertTriangle, Radio, Eye, Link2 } from 'lucide-react';
import type { JointResult } from '../types';
import { ConfidenceBar } from './ConfidenceBar';
import { TraceTimeline } from './TraceTimeline';

interface JointAnalysisPanelProps {
  joint: JointResult;
}

export function JointAnalysisPanel({ joint }: JointAnalysisPanelProps) {
  const confidencePct = Math.round(joint.confidence * 100);

  return (
    <div className="space-y-4 animate-fade-in-up">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg">🔗</span>
          <h3 className="text-sm font-semibold text-text-primary">Joint Optical + SAR Analysis</h3>
        </div>
        <span className="text-[10px] font-mono text-text-muted">
          {joint.models_used.join(' + ')}
        </span>
      </div>

      {/* Evidence reliability */}
      <div className="bg-bg-card border border-border rounded-lg p-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-text-secondary flex items-center gap-1.5">
            <Shield className="w-3 h-3" />
            Evidence Reliability
          </span>
          <span className="text-sm font-bold text-accent-teal">{confidencePct}%</span>
        </div>
        <ConfidenceBar confidence={joint.confidence} width="w-full" />
        <p className="text-[11px] text-text-muted mt-2 leading-relaxed">
          {joint.confidence_reasoning}
        </p>
      </div>

      {/* Evidence cards */}
      <div className="grid grid-cols-2 gap-2">
        {/* Optical evidence */}
        <div className="bg-bg-card border border-border rounded-lg p-3">
          <div className="flex items-center gap-1.5 mb-2">
            <Eye className="w-3 h-3 text-accent-blue" />
            <span className="text-[11px] font-semibold text-text-secondary">Optical</span>
          </div>
          {joint.optical?.success ? (
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-green" />
                <span className="text-[11px] text-text-secondary">EarthDial completed</span>
              </div>
              <span className="text-[10px] font-mono text-text-muted">
                {joint.optical.elapsed_s.toFixed(1)}s inference
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-accent-coral" />
              <span className="text-[11px] text-accent-coral">
                {joint.optical?.error || 'Failed'}
              </span>
            </div>
          )}
        </div>

        {/* SAR evidence */}
        <div className="bg-bg-card border border-border rounded-lg p-3">
          <div className="flex items-center gap-1.5 mb-2">
            <Radio className="w-3 h-3 text-accent-teal" />
            <span className="text-[11px] font-semibold text-text-secondary">SAR</span>
          </div>
          {joint.sar?.success ? (
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-green" />
                <span className="text-[11px] text-text-secondary">
                  {joint.sar.num_detections} vessel{joint.sar.num_detections !== 1 ? 's' : ''} detected
                </span>
              </div>
              <span className="text-[10px] font-mono text-text-muted">
                {joint.sar.inference_time_ms.toFixed(0)}ms
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-accent-coral" />
              <span className="text-[11px] text-accent-coral">
                {joint.sar?.error || 'Failed'}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Joint interpretation */}
      <div className="bg-bg-card border border-border rounded-lg p-3">
        <div className="flex items-center gap-1.5 mb-2">
          <Link2 className="w-3 h-3 text-accent-purple" />
          <span className="text-[11px] font-semibold text-text-secondary">Joint Interpretation</span>
        </div>
        <p className="text-xs text-text-secondary leading-relaxed">
          {joint.joint_answer || 'No interpretation generated'}
        </p>
      </div>

      {/* Execution trace */}
      {joint.trace.length > 0 && (
        <TraceTimeline steps={joint.trace} totalMs={joint.total_ms} />
      )}

      {/* Capabilities note */}
      <div className="bg-bg-card/50 border border-accent-amber/20 rounded-lg p-3">
        <div className="flex items-start gap-2">
          <AlertTriangle className="w-3.5 h-3.5 text-accent-amber mt-0.5 shrink-0" />
          <div>
            <p className="text-[11px] font-medium text-accent-amber mb-1">SAR Capability Limitations</p>
            <p className="text-[10px] text-text-muted leading-relaxed">
              The SAR specialist detects maritime vessels only. It cannot detect buildings, land cover,
              vegetation, or infrastructure changes. Results combine what both sensors can observe.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
