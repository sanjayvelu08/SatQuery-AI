import { Clock, Trash2 } from 'lucide-react';
import type { HistoryEntry } from '../types';
import { IntentBadge } from './IntentBadge';
import { truncate } from '../lib/utils';

interface HistoryPanelProps {
  entries: HistoryEntry[];
  onRevisit: (entry: HistoryEntry) => void;
  onClear: () => void;
}

export function HistoryPanel({ entries, onRevisit, onClear }: HistoryPanelProps) {
  if (entries.length === 0) {
    return (
      <div className="px-6 py-3 text-sm text-text-muted">
        <Clock className="w-4 h-4 inline mr-2 opacity-50" />
        No queries yet
      </div>
    );
  }

  return (
    <div className="border-t border-border bg-bg-secondary/50">
      <div className="flex items-center justify-between px-6 py-2.5">
        <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider">
          Query History ({entries.length})
        </h3>
        <button
          onClick={onClear}
          className="text-xs text-text-muted hover:text-accent-coral transition-colors flex items-center gap-1"
        >
          <Trash2 className="w-3 h-3" />
          Clear
        </button>
      </div>
      <div className="px-6 pb-3 flex flex-wrap gap-2">
        {[...entries].reverse().map((entry) => (
          <button
            key={entry.id}
            onClick={() => onRevisit(entry)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-border bg-bg-card/50 hover:bg-bg-card-hover hover:border-border-light transition-all text-left group"
          >
            <IntentBadge intent={entry.intent} size="sm" />
            <span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors">
              {truncate(entry.query, 40)}
            </span>
            <span className="text-[10px] text-text-muted font-mono">
              {entry.elapsed_s > 0 ? `${entry.elapsed_s.toFixed(1)}s` : '⚡'}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
