import { Satellite } from 'lucide-react';

interface HeaderProps {
  onAbout: () => void;
  demoCount: number;
}

export function Header({ onAbout, demoCount }: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-5 py-2.5 border-b border-border/60 bg-bg-secondary/90 backdrop-blur-md sticky top-0 z-50">
      {/* Left: Logo + Brand */}
      <div className="flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-teal to-accent-teal-hover flex items-center justify-center shadow-md shadow-accent-teal/15">
          <Satellite className="w-4 h-4 text-bg-primary" strokeWidth={2.5} />
        </div>
        <div>
          <h1 className="text-[15px] font-bold tracking-tight text-text-primary leading-none">
            SatQuery AI
          </h1>
          <p className="text-[10px] text-text-muted leading-none mt-0.5">
            ISRO SIH26167 · Remote Sensing
          </p>
        </div>
      </div>

      {/* Right: Info + Nav */}
      <div className="flex items-center gap-3">
        <span className="hidden md:inline text-[10px] text-text-muted font-mono bg-bg-card/60 px-2 py-0.5 rounded border border-border/50">
          EarthDial 4B · YOLOv8 SAR
        </span>
        {demoCount > 0 && (
          <span className="text-[10px] font-medium text-accent-teal bg-accent-teal/10 px-2 py-0.5 rounded-full border border-accent-teal/20">
            {demoCount} demos
          </span>
        )}
        <button
          onClick={onAbout}
          className="text-xs font-medium text-text-secondary hover:text-accent-teal transition-colors"
        >
          About
        </button>
        <a
          href="https://github.com/sanjayvelu08/SatQuery-AI"
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs font-medium text-text-muted hover:text-text-primary transition-colors"
        >
          GitHub
        </a>
      </div>
    </header>
  );
}
