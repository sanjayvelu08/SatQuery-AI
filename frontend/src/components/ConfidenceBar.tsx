interface ConfidenceBarProps {
  confidence: number;
  width?: string;
}

export function ConfidenceBar({ confidence, width = 'w-24' }: ConfidenceBarProps) {
  const pct = Math.round(confidence * 100);

  let barColor = 'bg-accent-coral';
  if (pct >= 70) barColor = 'bg-gradient-to-r from-accent-teal to-accent-green';
  else if (pct >= 30) barColor = 'bg-gradient-to-r from-accent-amber to-accent-amber';

  return (
    <div className={`flex items-center gap-2 ${width}`}>
      <div className="flex-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
        <div
          className={`h-full rounded-full ${barColor} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[11px] font-mono text-text-secondary w-10 text-right">
        {pct}%
      </span>
    </div>
  );
}
