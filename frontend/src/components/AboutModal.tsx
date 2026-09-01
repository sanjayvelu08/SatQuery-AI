import { X, ExternalLink } from 'lucide-react';

interface AboutModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function AboutModal({ isOpen, onClose }: AboutModalProps) {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      onClick={onClose}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in-up" />

      {/* Modal */}
      <div
        className="relative bg-bg-secondary border border-border rounded-xl max-w-2xl w-full max-h-[85vh] overflow-y-auto shadow-2xl animate-scale-in"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-text-muted hover:text-text-primary transition-colors z-10"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="p-7">
          {/* Header */}
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-accent-teal to-accent-teal-hover flex items-center justify-center text-lg">
              🛰️
            </div>
            <div>
              <h2 className="text-lg font-bold text-text-primary tracking-tight">SatQuery AI</h2>
              <p className="text-xs text-text-secondary">
                Remote Sensing Vision-Language Assistant · ISRO SIH 2026
              </p>
            </div>
          </div>

          {/* Architecture */}
          <section className="mb-6">
            <div className="section-label mb-3">System Architecture</div>
            <div className="bg-bg-primary/60 rounded-lg p-5 font-mono text-xs text-text-secondary space-y-2">
              <div className="flex items-center justify-center gap-2 text-text-primary">
                <span className="px-3 py-1 rounded bg-accent-teal/15 border border-accent-teal/30">
                  Image Upload
                </span>
                <span className="text-accent-teal">→</span>
                <span className="px-3 py-1 rounded bg-accent-teal/15 border border-accent-teal/30">
                  Keyword Router
                </span>
                <span className="text-accent-teal">→</span>
                <span className="px-3 py-1 rounded bg-accent-teal/15 border border-accent-teal/30">
                  Model Selection
                </span>
              </div>
              <div className="flex justify-center text-accent-teal text-lg">↓</div>
              <div className="flex items-center justify-center gap-3 text-text-primary">
                <span className="px-3 py-1 rounded bg-accent-green/15 border border-accent-green/30">
                  EarthDial 4B
                </span>
                <span className="px-3 py-1 rounded bg-accent-amber/15 border border-accent-amber/30">
                  YOLOv8 SAR
                </span>
              </div>
              <div className="flex justify-center text-accent-teal text-lg">↓</div>
              <div className="flex items-center justify-center">
                <span className="px-3 py-1 rounded bg-accent-teal/15 border border-accent-teal/30 text-text-primary">
                  Results + Visual Evidence
                </span>
              </div>
            </div>
          </section>

          {/* Models */}
          <section className="mb-6">
            <div className="section-label mb-3">Models</div>
            <div className="rounded-lg border border-border overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-accent-teal/8">
                    <th className="px-4 py-2 text-left text-[10px] font-semibold text-text-secondary uppercase tracking-wider">
                      Model
                    </th>
                    <th className="px-4 py-2 text-left text-[10px] font-semibold text-text-secondary uppercase tracking-wider">
                      Purpose
                    </th>
                    <th className="px-4 py-2 text-left text-[10px] font-semibold text-text-secondary uppercase tracking-wider hidden sm:table-cell">
                      VRAM
                    </th>
                    <th className="px-4 py-2 text-left text-[10px] font-semibold text-text-secondary uppercase tracking-wider hidden sm:table-cell">
                      Speed
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  <tr>
                    <td className="px-4 py-2.5 font-medium text-text-primary">EarthDial 4B</td>
                    <td className="px-4 py-2.5 text-text-secondary">Optical VLM</td>
                    <td className="px-4 py-2.5 text-text-muted font-mono text-xs hidden sm:table-cell">2.85 GB</td>
                    <td className="px-4 py-2.5 text-text-muted font-mono text-xs hidden sm:table-cell">50-260s</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2.5 font-medium text-text-primary">YOLOv8n</td>
                    <td className="px-4 py-2.5 text-text-secondary">SAR vessel detection</td>
                    <td className="px-4 py-2.5 text-text-muted font-mono text-xs hidden sm:table-cell">21 MB</td>
                    <td className="px-4 py-2.5 text-text-muted font-mono text-xs hidden sm:table-cell">3-7ms</td>
                  </tr>
                  <tr>
                    <td className="px-4 py-2.5 font-medium text-text-primary">Router</td>
                    <td className="px-4 py-2.5 text-text-secondary">Intent classifier</td>
                    <td className="px-4 py-2.5 text-text-muted font-mono text-xs hidden sm:table-cell">0 MB</td>
                    <td className="px-4 py-2.5 text-text-muted font-mono text-xs hidden sm:table-cell">&lt;1ms</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          {/* Capabilities */}
          <section className="mb-6">
            <div className="section-label mb-3">Capabilities</div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 text-sm">
              {[
                { label: 'Optical image captioning', status: 'ready' as const },
                { label: 'Visual question answering', status: 'ready' as const },
                { label: 'Feature detection + grounding', status: 'ready' as const },
                { label: 'Scene classification', status: 'ready' as const },
                { label: 'SAR maritime vessel detection', status: 'ready' as const },
                { label: 'Change detection', status: 'planned' as const },
              ].map((cap) => (
                <div key={cap.label} className="flex items-center gap-2 text-text-secondary py-1">
                  <span className={cap.status === 'ready' ? 'text-accent-green' : 'text-text-muted'}>
                    {cap.status === 'ready' ? '✓' : '○'}
                  </span>
                  {cap.label}
                </div>
              ))}
            </div>
          </section>

          {/* Tech stack */}
          <section className="mb-5">
            <div className="section-label mb-3">Technical Details</div>
            <div className="text-sm text-text-secondary space-y-1">
              <p>Hardware: NVIDIA RTX 3050 (4 GB)</p>
              <p>Backend: Python 3.12 · PyTorch 2.5.1 · HuggingFace Transformers</p>
              <p>Frontend: React 18 · Vite · TypeScript · Tailwind CSS</p>
            </div>
          </section>

          {/* Footer */}
          <div className="pt-3 border-t border-border flex items-center justify-between text-xs text-text-muted">
            <span>Built for SIH 2026</span>
            <a
              href="https://github.com/sanjayvelu08/SatQuery-AI"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-accent-teal hover:text-accent-teal-hover transition-colors"
            >
              View on GitHub
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
