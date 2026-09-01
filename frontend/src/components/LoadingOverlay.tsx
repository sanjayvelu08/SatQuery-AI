import { useEffect, useState } from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingOverlayProps {
  elapsedSeconds: number;
  onCancel: () => void;
}

const STAGES = [
  { label: 'Route query', instant: true },
  { label: 'Analyzing imagery', instant: false },
  { label: 'Generating response', instant: false },
];

export function LoadingOverlay({ elapsedSeconds, onCancel }: LoadingOverlayProps) {
  const [activeStage, setActiveStage] = useState(0);
  const [completedStages, setCompletedStages] = useState<number[]>([]);

  useEffect(() => {
    // Stage progression based on elapsed time
    const timer = setTimeout(() => {
      if (elapsedSeconds < 1 && !completedStages.includes(0)) {
        setActiveStage(0);
        setCompletedStages((prev) => [...prev, 0]);
        setTimeout(() => setActiveStage(1), 500);
      } else if (elapsedSeconds >= 2 && !completedStages.includes(1)) {
        setCompletedStages((prev) => [...prev, 1]);
        setActiveStage(2);
      }
    }, 100);
    return () => clearTimeout(timer);
  }, [elapsedSeconds, completedStages]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
  };

  return (
    <div className="flex flex-col items-center justify-center py-16 px-8 animate-fade-in-up">
      {/* Orbital animation */}
      <div className="relative w-20 h-20 mb-6">
        <div className="absolute inset-0 rounded-full border-2 border-border" />
        <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-accent-teal animate-orbital-spin" />
        <div className="absolute inset-3 rounded-full border border-border-light" />
        <div className="absolute inset-0 flex items-center justify-center">
          <Loader2 className="w-6 h-6 text-accent-teal animate-spin" />
        </div>
      </div>

      {/* Loading text */}
      <p className="text-text-secondary text-sm mb-6">
        Analyzing satellite imagery...
      </p>

      {/* Progress stages */}
      <div className="flex items-center gap-2 mb-6">
        {STAGES.map((_stage, i) => (
          <div key={i} className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full border-2 transition-all duration-300 ${
                completedStages.includes(i)
                  ? 'bg-accent-teal border-accent-teal'
                  : activeStage === i
                    ? 'border-accent-teal animate-pulse-glow'
                    : 'border-border'
              }`}
            />
            {i < STAGES.length - 1 && (
              <div
                className={`w-8 h-0.5 transition-colors duration-300 ${
                  completedStages.includes(i) ? 'bg-accent-teal' : 'bg-border'
                }`}
              />
            )}
          </div>
        ))}
      </div>

      {/* Stage labels */}
      <div className="flex items-center gap-4 mb-6 text-xs text-text-muted">
        {STAGES.map((s, i) => (
          <span
            key={i}
            className={`transition-colors duration-300 ${
              activeStage === i ? 'text-accent-teal' : ''
            }`}
          >
            {s.label}
          </span>
        ))}
      </div>

      {/* Elapsed time */}
      <div className="font-mono text-2xl text-text-primary mb-2 tabular-nums">
        {formatTime(elapsedSeconds)}
      </div>
      <p className="text-[11px] text-text-muted mb-6">elapsed</p>

      {/* Cancel button */}
      <button
        onClick={onCancel}
        className="px-4 py-2 text-sm text-text-secondary border border-border rounded-md hover:bg-bg-card-hover hover:text-text-primary transition-colors"
      >
        Cancel
      </button>
    </div>
  );
}
