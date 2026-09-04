import { Scan } from 'lucide-react';
import type { IntentType, ChangeResult, JointResult } from '../types';

interface VisualEvidenceProps {
  annotatedImageUrl: string | null;
  originalImageUrl: string;
  intent: IntentType;
  changeResult?: ChangeResult | null;
  jointResult?: JointResult | null;
}

export function VisualEvidence({
  annotatedImageUrl,
  originalImageUrl,
  intent,
  changeResult,
  jointResult,
}: VisualEvidenceProps) {
  // Change detection overlay
  if (intent === 'change' && changeResult?.overlay_url) {
    return (
      <div className="space-y-2 animate-scale-in">
        <div className="rounded-lg overflow-hidden border border-accent-teal/30">
          <img
            src={changeResult.overlay_url}
            alt="Change detection overlay"
            className="w-full rounded-lg"
          />
        </div>
        {changeResult.bbox_url && (
          <div className="rounded-lg overflow-hidden border border-border">
            <img
              src={changeResult.bbox_url}
              alt="Change bounding boxes"
              className="w-full rounded-lg"
            />
          </div>
        )}
      </div>
    );
  }

  // Joint analysis — show the exact OPTICAL|SAR composite the joint
  // interpretation used, plus the source optical image
  if (intent === 'joint_analysis' && jointResult) {
    return (
      <div className="space-y-2 animate-scale-in">
        {jointResult.composite_url && (
          <div className="rounded-lg overflow-hidden border border-accent-teal/30">
            <img
              src={jointResult.composite_url}
              alt="OPTICAL | SAR composite"
              className="w-full rounded-lg"
            />
            <div className="px-2 py-1 bg-bg-card text-[10px] text-text-muted text-center">
              OPTICAL | SAR composite — evidence used by the joint interpretation
            </div>
          </div>
        )}
        {originalImageUrl && (
          <div className="rounded-lg overflow-hidden border border-border">
            <img
              src={originalImageUrl}
              alt="Optical image"
              className="w-full rounded-lg opacity-90"
            />
            <div className="px-2 py-1 bg-bg-card text-[10px] text-text-muted text-center">
              Optical Image
            </div>
          </div>
        )}
      </div>
    );
  }

  // SAR / detection / grounding — annotated image
  const showAnnotated =
    !!annotatedImageUrl &&
    (intent === 'detect' || intent === 'grounding' || intent === 'sar');

  if (showAnnotated) {
    return (
      <div className="animate-scale-in rounded-lg overflow-hidden border border-accent-teal/30">
        <img
          src={annotatedImageUrl}
          alt="Annotated analysis result"
          className="w-full rounded-lg"
        />
      </div>
    );
  }

  // Default — original image
  if (originalImageUrl) {
    return (
      <div className="animate-scale-in rounded-lg overflow-hidden border border-border">
        <img
          src={originalImageUrl}
          alt="Uploaded satellite image"
          className="w-full rounded-lg opacity-90"
        />
      </div>
    );
  }

  // Empty state
  return (
    <div className="flex flex-col items-center justify-center py-10 border border-dashed border-border rounded-lg bg-bg-primary/40">
      <div className="w-12 h-12 rounded-full bg-bg-card border border-border flex items-center justify-center mb-3">
        <Scan className="w-5 h-5 text-text-muted/40" />
      </div>
      <p className="text-xs text-text-muted text-center leading-relaxed">
        Upload an image to begin analysis
      </p>
    </div>
  );
}
