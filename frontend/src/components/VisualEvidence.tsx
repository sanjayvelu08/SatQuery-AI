import { Scan } from 'lucide-react';
import type { IntentType } from '../types';

interface VisualEvidenceProps {
  annotatedImageUrl: string | null;
  originalImageUrl: string;
  intent: IntentType;
}

export function VisualEvidence({
  annotatedImageUrl,
  originalImageUrl,
  intent,
}: VisualEvidenceProps) {
  const hasAnnotated = !!annotatedImageUrl;
  const showAnnotated = hasAnnotated && (intent === 'detect' || intent === 'grounding' || intent === 'sar');

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

  return (
    <div className="flex flex-col items-center justify-center py-10 border border-dashed border-border rounded-lg bg-bg-primary/40">
      <div className="w-12 h-12 rounded-full bg-bg-card border border-border flex items-center justify-center mb-3">
        <Scan className="w-5 h-5 text-text-muted/40" />
      </div>
      <p className="text-xs text-text-muted text-center leading-relaxed">
        Bounding boxes appear here for
        <br />
        detection and grounding queries
      </p>
    </div>
  );
}
