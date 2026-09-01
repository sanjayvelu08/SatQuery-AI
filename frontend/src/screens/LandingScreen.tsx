import { Upload, ArrowRight } from 'lucide-react';
import { OrbitalSvg } from '../components/OrbitalSvg';
import { CAPABILITY_CARDS } from '../lib/constants';

interface LandingScreenProps {
  onUpload: (file: File) => void;
}

export function LandingScreen({ onUpload }: LandingScreenProps) {
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      onUpload(file);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center px-6 py-10 ambient-glow">
      {/* Hero */}
      <div className="text-center max-w-2xl mb-10 animate-fade-in-up">
        <div className="flex justify-center mb-5">
          <OrbitalSvg size={90} />
        </div>
        <h1 className="text-4xl font-bold tracking-tight mb-3 bg-gradient-to-r from-accent-teal via-accent-cyan to-accent-teal bg-clip-text text-transparent leading-tight">
          Satellite Intelligence
          <br />
          At Your Command
        </h1>
        <p className="text-text-secondary text-base max-w-md mx-auto leading-relaxed">
          Upload a satellite image. Ask a natural-language question.
          <br />
          Get AI-powered analysis in seconds.
        </p>
      </div>

      {/* Upload zone */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        className="w-full max-w-lg mb-10"
      >
        <label className="flex flex-col items-center gap-3 p-7 border-2 border-dashed border-border-light rounded-xl bg-bg-card/40 hover:border-accent-teal/50 hover:bg-accent-teal/5 transition-all cursor-pointer group">
          <div className="w-12 h-12 rounded-full bg-accent-teal/10 flex items-center justify-center group-hover:bg-accent-teal/15 transition-colors">
            <Upload className="w-5 h-5 text-accent-teal" />
          </div>
          <div className="text-center">
            <span className="text-sm font-semibold text-text-primary block">
              Upload Satellite Image
            </span>
            <p className="text-xs text-text-muted mt-1">
              Drag & drop or click to browse · Sentinel-2, Landsat, SAR
            </p>
          </div>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
        </label>
      </div>

      {/* Capability cards */}
      <div className="w-full max-w-3xl mb-10">
        <div className="section-label justify-center mb-5">
          Capabilities
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 stagger-children">
          {CAPABILITY_CARDS.map((card) => (
            <div
              key={card.intent}
              className="bg-bg-card/60 border border-border rounded-xl p-4 text-center hover:border-border-light hover:bg-bg-card-hover transition-all group card-glow"
            >
              <div className="text-3xl mb-2 group-hover:animate-subtle-float">
                {card.icon}
              </div>
              <h3 className="text-sm font-semibold text-text-primary mb-1">
                {card.title}
              </h3>
              <p className="text-[11px] text-text-muted leading-snug">
                {card.desc}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* How it works */}
      <div className="w-full max-w-2xl">
        <div className="section-label justify-center mb-5">
          How It Works
        </div>
        <div className="flex items-center justify-between gap-2">
          {['Upload', 'AI Routes', 'Inference', 'Results'].map((step, i) => (
            <div key={step} className="flex items-center gap-2">
              <div className="flex flex-col items-center gap-2">
                <div className="w-11 h-11 rounded-full border border-border-light bg-bg-card flex items-center justify-center text-sm font-bold text-accent-teal shadow-sm">
                  {i + 1}
                </div>
                <span className="text-[11px] font-medium text-text-secondary">{step}</span>
              </div>
              {i < 3 && (
                <ArrowRight className="w-4 h-4 text-text-muted/40 mt-[-18px]" />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
