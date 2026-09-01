import { ChevronDown } from 'lucide-react';
import type { DemoScenario } from '../types';

interface DemoSelectorProps {
  demos: DemoScenario[];
  selected: string;
  onSelect: (name: string) => void;
}

export function DemoSelector({ demos, selected, onSelect }: DemoSelectorProps) {
  return (
    <div className="relative">
      <select
        value={selected}
        onChange={(e) => onSelect(e.target.value)}
        className="w-full appearance-none bg-bg-card border border-border rounded-lg px-4 py-2.5 pr-10 text-sm text-text-primary focus:outline-none focus:border-accent-teal focus:ring-1 focus:ring-accent-teal/30 transition-colors cursor-pointer"
      >
        <option value="">📋 Select a demo scenario...</option>
        {demos.map((demo) => (
          <option key={demo.name} value={demo.name}>
            {demo.name}
          </option>
        ))}
      </select>
      <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted pointer-events-none" />
    </div>
  );
}
