import type { IntentType } from '../types';
import { INTENT_COLORS, INTENT_LABELS, INTENT_ICONS } from '../lib/constants';
import { cn } from '../lib/utils';

interface IntentBadgeProps {
  intent: IntentType;
  size?: 'sm' | 'md';
}

export function IntentBadge({ intent, size = 'md' }: IntentBadgeProps) {
  const color = INTENT_COLORS[intent];
  const label = INTENT_LABELS[intent];
  const icon = INTENT_ICONS[intent];

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full font-medium uppercase tracking-wider',
        size === 'sm' ? 'text-[10px] px-2 py-0.5' : 'text-[11px] px-2.5 py-1',
      )}
      style={{
        color,
        backgroundColor: `${color}18`,
        border: `1px solid ${color}30`,
      }}
    >
      <span>{icon}</span>
      <span>{label}</span>
    </span>
  );
}
