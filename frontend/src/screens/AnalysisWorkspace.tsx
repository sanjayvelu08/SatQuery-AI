import { useState, useRef, useCallback, useEffect } from 'react';
import {
  Upload, X, Search, Loader2, AlertTriangle, ChevronDown,
  Image as ImageIcon, MessageSquare, BarChart3, GitCompare, Radio,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type {
  AnalyzeResponse, DemoScenario, HistoryEntry, IntentType, AnalysisMode,
} from '../types';
import { INTENT_COLORS, INTENT_LABELS, INTENT_ICONS, MODEL_LABELS } from '../lib/constants';
import { formatTiming } from '../lib/utils';
import { getAnnotatedUrl } from '../api/client';
import { IntentBadge } from '../components/IntentBadge';
import { VisualEvidence } from '../components/VisualEvidence';
import { LoadingOverlay } from '../components/LoadingOverlay';
import { DetectionTable } from '../components/DetectionTable';
import { ChangeResultPanel } from '../components/ChangeResultPanel';
import { JointAnalysisPanel } from '../components/JointAnalysisPanel';
import { TraceTimeline } from '../components/TraceTimeline';

const MODE_CONFIG: Record<AnalysisMode, {
  label: string;
  icon: React.ReactNode;
  description: string;
  needsT2: boolean;
  needsSar: boolean;
}> = {
  single: {
    label: 'Single Image',
    icon: <ImageIcon className="w-3.5 h-3.5" />,
    description: 'Caption, VQA, detection, classification',
    needsT2: false,
    needsSar: false,
  },
  change: {
    label: 'Change Detection',
    icon: <GitCompare className="w-3.5 h-3.5" />,
    description: 'Compare before/after satellite images',
    needsT2: true,
    needsSar: false,
  },
  joint: {
    label: 'Optical + SAR',
    icon: <Radio className="w-3.5 h-3.5" />,
    description: 'Fused optical and radar analysis',
    needsT2: false,
    needsSar: true,
  },
};

interface AnalysisWorkspaceProps {
  onBackToLanding: () => void;
  demos: DemoScenario[];
  onAnalyze: (
    image: File,
    query: string,
    imageT2?: File | null,
    imageSar?: File | null,
  ) => Promise<AnalyzeResponse>;
  onAnalyzeDemo: (demo: DemoScenario) => Promise<AnalyzeResponse>;
}

export function AnalysisWorkspace({
  onBackToLanding,
  demos,
  onAnalyze,
  onAnalyzeDemo,
}: AnalysisWorkspaceProps) {
  const [mode, setMode] = useState<AnalysisMode>('single');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageT2File, setImageT2File] = useState<File | null>(null);
  const [imageSarFile, setImageSarFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState('');
  const [imageT2Preview, setImageT2Preview] = useState('');
  const [imageSarPreview, setImageSarPreview] = useState('');
  const [query, setQuery] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [selectedDemo, setSelectedDemo] = useState('');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [showDemos, setShowDemos] = useState(false);
  const [showModeMenu, setShowModeMenu] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const fileT2Ref = useRef<HTMLInputElement>(null);
  const fileSarRef = useRef<HTMLInputElement>(null);

  // Elapsed timer
  useEffect(() => {
    if (!isAnalyzing) return;
    setElapsedSeconds(0);
    const interval = setInterval(() => setElapsedSeconds((p) => p + 1), 1000);
    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const clearAll = useCallback(() => {
    setImageFile(null);
    setImageT2File(null);
    setImageSarFile(null);
    setImagePreview('');
    setImageT2Preview('');
    setImageSarPreview('');
    setQuery('');
    setResult(null);
    setError(null);
    setSelectedDemo('');
  }, []);

  const handleModeChange = useCallback((newMode: AnalysisMode) => {
    setMode(newMode);
    setShowModeMenu(false);
    // Clear T2/SAR when switching modes
    if (newMode !== 'change') {
      setImageT2File(null);
      setImageT2Preview('');
    }
    if (newMode !== 'joint') {
      setImageSarFile(null);
      setImageSarPreview('');
    }
  }, []);

  const handleFile = useCallback((file: File, which: 'main' | 't2' | 'sar') => {
    const preview = URL.createObjectURL(file);
    if (which === 'main') {
      setImageFile(file);
      setImagePreview(preview);
    } else if (which === 't2') {
      setImageT2File(file);
      setImageT2Preview(preview);
    } else {
      setImageSarFile(file);
      setImageSarPreview(preview);
    }
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = (e: React.DragEvent, which: 'main' | 't2' | 'sar') => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file, which);
  };

  const handleAnalyze = async () => {
    if (!imageFile || !query.trim()) return;
    setIsAnalyzing(true);
    setError(null);
    try {
      const response = await onAnalyze(
        imageFile,
        query.trim(),
        mode === 'change' ? imageT2File : null,
        mode === 'joint' ? imageSarFile : null,
      );
      setResult(response);
      const entry: HistoryEntry = {
        id: Date.now().toString(),
        timestamp: Date.now(),
        query: query.trim(),
        intent: response.intent,
        model_used: response.model_used,
        elapsed_s: response.elapsed_total_s,
        supported: response.supported,
        result: response,
        imagePreview,
        mode,
      };
      setHistory((prev) => [...prev, entry].slice(-5));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDemoSelect = async (demoName: string) => {
    if (!demoName) return;
    const demo = demos.find((d) => d.name === demoName);
    if (!demo) return;
    setSelectedDemo(demoName);
    setImagePreview(demo.image_url);
    setQuery(demo.query);
    setIsAnalyzing(true);
    setError(null);
    try {
      const response = await onAnalyzeDemo(demo);
      setResult(response);
      const entry: HistoryEntry = {
        id: Date.now().toString(),
        timestamp: Date.now(),
        query: demo.query,
        intent: response.intent,
        model_used: response.model_used,
        elapsed_s: response.elapsed_total_s,
        supported: response.supported,
        result: response,
        imagePreview: demo.image_url,
        mode: 'single',
      };
      setHistory((prev) => [...prev, entry].slice(-5));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Demo failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const annotatedUrl = result ? getAnnotatedUrl(result.annotated_image_url) : null;
  const cfg = MODE_CONFIG[mode];

  const canAnalyze =
    imageFile &&
    query.trim() &&
    !isAnalyzing &&
    (mode !== 'change' || imageT2File) &&
    (mode !== 'joint' || imageSarFile);

  const renderAnswerMarkdown = (answer: string) => (
    <div className="prose prose-invert prose-sm max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="overflow-x-auto my-3 rounded-lg border border-border">
              <table className="w-full text-sm">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="px-3 py-2 text-left text-[11px] font-semibold text-text-secondary uppercase bg-accent-teal/8 border-b border-border">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-3 py-2 text-sm text-text-primary border-b border-border/50">
              {children}
            </td>
          ),
          p: ({ children }) => (
            <p className="text-sm text-text-secondary leading-relaxed mb-2">{children}</p>
          ),
          strong: ({ children }) => (
            <strong className="text-text-primary font-semibold">{children}</strong>
          ),
          em: ({ children }) => (
            <em className="text-text-muted text-xs">{children}</em>
          ),
          h3: ({ children }) => (
            <h3 className="text-sm font-semibold text-text-primary mt-4 mb-2">{children}</h3>
          ),
          ul: ({ children }) => (
            <ul className="text-sm text-text-secondary space-y-1 mb-2 ml-4 list-disc">{children}</ul>
          ),
          li: ({ children }) => <li>{children}</li>,
          hr: () => <hr className="border-border my-3" />,
        }}
      >
        {answer}
      </ReactMarkdown>
    </div>
  );

  return (
    <div className="flex-1 flex flex-col">
      {/* Top bar: demos + mode selector */}
      <div className="px-5 py-2.5 border-b border-border bg-bg-secondary/50">
        <div className="flex items-center gap-3">
          <button
            onClick={onBackToLanding}
            className="text-sm text-text-muted hover:text-accent-teal transition-colors"
          >
            ← Home
          </button>

          {/* Mode selector */}
          <div className="relative">
            <button
              onClick={() => setShowModeMenu(!showModeMenu)}
              className="flex items-center gap-2 bg-bg-card border border-border rounded-lg px-3 py-2 text-sm text-text-secondary hover:border-border-light hover:bg-bg-card-hover transition-all"
            >
              {cfg.icon}
              <span className="text-xs font-medium">{cfg.label}</span>
              <ChevronDown className={`w-3 h-3 transition-transform ${showModeMenu ? 'rotate-180' : ''}`} />
            </button>
            {showModeMenu && (
              <div className="absolute top-full left-0 mt-1 bg-bg-card border border-border rounded-lg shadow-xl shadow-black/20 z-20 w-56">
                {(Object.keys(MODE_CONFIG) as AnalysisMode[]).map((m) => (
                  <button
                    key={m}
                    onClick={() => handleModeChange(m)}
                    className={`w-full text-left px-3 py-2.5 text-sm transition-colors border-b border-border/30 last:border-0 ${
                      m === mode
                        ? 'bg-accent-teal/10 text-accent-teal'
                        : 'text-text-secondary hover:bg-bg-card-hover'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {MODE_CONFIG[m].icon}
                      <span className="font-medium text-xs">{MODE_CONFIG[m].label}</span>
                    </div>
                    <p className="text-[10px] text-text-muted mt-0.5 ml-5.5">
                      {MODE_CONFIG[m].description}
                    </p>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Demo selector */}
          <div className="relative flex-1 max-w-md">
            <button
              onClick={() => setShowDemos(!showDemos)}
              className="w-full flex items-center justify-between bg-bg-card border border-border rounded-lg px-3 py-2 text-sm text-text-secondary hover:border-border-light hover:bg-bg-card-hover transition-all"
            >
              <span className="flex items-center gap-2">
                {selectedDemo
                  ? demos.find((d) => d.name === selectedDemo)?.name || selectedDemo
                  : 'Quick Demo — pre-computed scenarios'}
              </span>
              <ChevronDown className={`w-4 h-4 transition-transform ${showDemos ? 'rotate-180' : ''}`} />
            </button>
            {showDemos && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-bg-card border border-border rounded-lg shadow-xl shadow-black/20 z-10 max-h-60 overflow-y-auto">
                {demos.map((demo) => (
                  <button
                    key={demo.name}
                    onClick={() => { setShowDemos(false); handleDemoSelect(demo.name); }}
                    className="w-full text-left px-4 py-2.5 text-sm text-text-secondary hover:bg-accent-teal/8 hover:text-text-primary transition-colors border-b border-border/30 last:border-0"
                  >
                    {demo.name}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main workspace */}
      <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
        {/* LEFT: Uploads + Query + Visual Evidence */}
        <div className="lg:w-1/2 flex flex-col border-r border-border overflow-y-auto">

          {/* Primary image upload */}
          <div className="p-4 border-b border-border">
            <div className="section-label mb-2">
              <ImageIcon className="w-3 h-3" />
              {mode === 'change' ? 'Earlier Image (T1)' : mode === 'joint' ? 'Optical Image' : 'Satellite Image'}
            </div>
            {imagePreview ? (
              <div className="relative rounded-lg overflow-hidden border border-border bg-bg-primary">
                <img src={imagePreview} alt="Primary image" className="w-full h-auto max-h-48 object-contain" />
                <button
                  onClick={() => { setImageFile(null); setImagePreview(''); }}
                  className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/50 backdrop-blur flex items-center justify-center text-white/70 hover:text-white hover:bg-black/70 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <label
                onDrop={(e) => handleDrop(e, 'main')}
                onDragOver={(e) => e.preventDefault()}
                className="flex flex-col items-center gap-2 p-6 border-2 border-dashed border-border rounded-lg bg-bg-primary/30 hover:border-accent-teal/40 hover:bg-accent-teal/3 transition-all cursor-pointer"
              >
                <Upload className="w-5 h-5 text-text-muted" />
                <span className="text-xs text-text-muted">Drag & drop or click to upload</span>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f, 'main'); }}
                  className="hidden"
                />
              </label>
            )}
          </div>

          {/* T2 upload (change detection mode) */}
          {cfg.needsT2 && (
            <div className="p-4 border-b border-border">
              <div className="section-label mb-2">
                <GitCompare className="w-3 h-3" />
                Later Image (T2)
              </div>
              {imageT2Preview ? (
                <div className="relative rounded-lg overflow-hidden border border-border bg-bg-primary">
                  <img src={imageT2Preview} alt="T2 image" className="w-full h-auto max-h-48 object-contain" />
                  <button
                    onClick={() => { setImageT2File(null); setImageT2Preview(''); }}
                    className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/50 backdrop-blur flex items-center justify-center text-white/70 hover:text-white hover:bg-black/70 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ) : (
                <label
                  onDrop={(e) => handleDrop(e, 't2')}
                  onDragOver={(e) => e.preventDefault()}
                  className="flex flex-col items-center gap-2 p-6 border-2 border-dashed border-border rounded-lg bg-bg-primary/30 hover:border-accent-teal/40 hover:bg-accent-teal/3 transition-all cursor-pointer"
                >
                  <Upload className="w-5 h-5 text-text-muted" />
                  <span className="text-xs text-text-muted">Upload the later satellite image</span>
                  <input
                    ref={fileT2Ref}
                    type="file"
                    accept="image/*"
                    onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f, 't2'); }}
                    className="hidden"
                  />
                </label>
              )}
            </div>
          )}

          {/* SAR upload (joint analysis mode) */}
          {cfg.needsSar && (
            <div className="p-4 border-b border-border">
              <div className="section-label mb-2">
                <Radio className="w-3 h-3" />
                SAR Image
              </div>
              {imageSarPreview ? (
                <div className="relative rounded-lg overflow-hidden border border-border bg-bg-primary">
                  <img src={imageSarPreview} alt="SAR image" className="w-full h-auto max-h-48 object-contain" />
                  <button
                    onClick={() => { setImageSarFile(null); setImageSarPreview(''); }}
                    className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/50 backdrop-blur flex items-center justify-center text-white/70 hover:text-white hover:bg-black/70 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ) : (
                <label
                  onDrop={(e) => handleDrop(e, 'sar')}
                  onDragOver={(e) => e.preventDefault()}
                  className="flex flex-col items-center gap-2 p-6 border-2 border-dashed border-border rounded-lg bg-bg-primary/30 hover:border-accent-teal/40 hover:bg-accent-teal/3 transition-all cursor-pointer"
                >
                  <Upload className="w-5 h-5 text-text-muted" />
                  <span className="text-xs text-text-muted">Upload the SAR radar image</span>
                  <input
                    ref={fileSarRef}
                    type="file"
                    accept="image/*"
                    onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f, 'sar'); }}
                    className="hidden"
                  />
                </label>
              )}
            </div>
          )}

          {/* Query input */}
          <div className="p-4 border-b border-border">
            <div className="section-label mb-2">
              <MessageSquare className="w-3 h-3" />
              Query
            </div>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={
                mode === 'change'
                  ? 'What changed between these two images?'
                  : mode === 'joint'
                    ? 'Analyze the optical and SAR images together'
                    : 'Describe this image · Are there buildings? · Detect ships in SAR'
              }
              rows={2}
              className="w-full bg-bg-input border border-border rounded-lg px-3 py-2.5 text-sm text-text-primary placeholder:text-text-muted/60 focus:outline-none focus:border-accent-teal focus:ring-1 focus:ring-accent-teal/30 transition-colors resize-none"
            />
            {/* Suggestion chips */}
            {imagePreview && !result && !isAnalyzing && (
              <div className="flex flex-wrap gap-1.5 mt-2">
                {(mode === 'change'
                  ? ['What changed here?', 'Detect new buildings', 'Compare these dates']
                  : mode === 'joint'
                    ? ['Identify vessels', 'Describe both images', 'What does SAR show?']
                    : ['Describe this image', 'Are there buildings?', 'Classify land use', 'Detect features']
                ).map((s) => (
                  <button
                    key={s}
                    onClick={() => setQuery(s)}
                    className="px-2.5 py-1 text-[11px] font-medium rounded-full border border-accent-teal/20 bg-accent-teal/5 text-accent-teal hover:bg-accent-teal/15 hover:border-accent-teal/40 transition-all"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
            {/* Action buttons */}
            <div className="flex gap-2 mt-3">
              <button
                onClick={handleAnalyze}
                disabled={!canAnalyze}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg font-semibold text-sm bg-gradient-to-r from-accent-teal to-accent-teal-hover text-bg-primary shadow-md shadow-accent-teal/20 hover:shadow-lg hover:shadow-accent-teal/30 hover:-translate-y-px active:translate-y-0 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:hover:shadow-md transition-all"
              >
                {isAnalyzing ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing...</>
                ) : (
                  <><Search className="w-4 h-4" /> Analyze</>
                )}
              </button>
              <button
                onClick={clearAll}
                className="px-4 py-2.5 rounded-lg text-sm font-medium text-text-secondary border border-border hover:bg-bg-card-hover hover:text-text-primary transition-colors"
              >
                Clear
              </button>
            </div>
          </div>

          {/* Visual Evidence */}
          <div className="p-4">
            <div className="section-label mb-2">
              <BarChart3 className="w-3 h-3" />
              Visual Evidence
            </div>
            <VisualEvidence
              annotatedImageUrl={annotatedUrl}
              originalImageUrl={imagePreview}
              intent={result?.intent || 'general'}
              changeResult={result?.change_result}
              jointResult={result?.joint_result}
            />
          </div>
        </div>

        {/* RIGHT: Results */}
        <div className="lg:w-1/2 flex flex-col overflow-y-auto">
          <div className="p-4 flex-1">
            <div className="section-label mb-3">
              <BarChart3 className="w-3 h-3" />
              Analysis Result
            </div>

            {isAnalyzing ? (
              <LoadingOverlay elapsedSeconds={elapsedSeconds} onCancel={() => setIsAnalyzing(false)} />
            ) : error ? (
              <div className="flex flex-col items-center gap-3 py-12 text-center animate-fade-in-up">
                <AlertTriangle className="w-10 h-10 text-accent-coral" />
                <p className="text-sm text-accent-coral font-medium">Analysis Error</p>
                <p className="text-sm text-text-secondary max-w-sm">{error}</p>
                <button
                  onClick={() => setError(null)}
                  className="text-sm text-accent-teal hover:text-accent-teal-hover transition-colors"
                >
                  Try again
                </button>
              </div>
            ) : result ? (
              <div className="animate-fade-in-up space-y-4">
                {/* Result header */}
                <div className="flex items-center justify-between">
                  <IntentBadge intent={result.intent} />
                  {result.supported ? (
                    <span className="text-xs text-accent-green flex items-center gap-1 font-medium">
                      Complete
                    </span>
                  ) : (
                    <span className="text-xs text-accent-amber flex items-center gap-1 font-medium">
                      {INTENT_LABELS[result.intent]}
                    </span>
                  )}
                </div>

                {/* Joint analysis panel */}
                {result.intent === 'joint_analysis' && result.joint_result ? (
                  <JointAnalysisPanel joint={result.joint_result} />
                ) : result.intent === 'change' && result.change_result ? (
                  <>
                    <ChangeResultPanel
                      change={result.change_result}
                      originalImageUrl={imagePreview}
                    />
                    {/* Semantic interpretation / pipeline answer */}
                    {result.supported && result.answer && renderAnswerMarkdown(result.answer)}
                  </>
                ) : result.supported && result.answer ? (
                  renderAnswerMarkdown(result.answer)
                ) : !result.supported ? (
                  <div className="py-6 text-center">
                    <AlertTriangle className="w-8 h-8 text-accent-amber mx-auto mb-3" />
                    <p className="text-sm text-text-secondary mb-2">{result.unsupported_reason}</p>
                    <p className="text-xs text-text-muted">
                      Try a different query type, such as describing the image or asking a question.
                    </p>
                  </div>
                ) : null}

                {/* SAR detections table */}
                {result.sar_result?.success && result.sar_result.detections.length > 0 && (
                  <DetectionTable detections={result.sar_result.detections} type="sar" />
                )}

                {/* Model + timing footer */}
                <div className="flex items-center justify-between pt-3 border-t border-border">
                  <span className="text-xs text-text-muted flex items-center gap-1.5">
                    {MODEL_LABELS[result.model_used] || result.model_used}
                  </span>
                  <span className="text-xs font-mono text-text-muted">
                    {formatTiming(result.elapsed_route_ms, result.elapsed_vlm_s, result.elapsed_total_s)}
                  </span>
                </div>

                {/* SAR VRAM */}
                {result.sar_result?.success && result.sar_result.gpu_vram_mb > 0 && (
                  <span className="text-[11px] font-mono text-text-muted">
                    GPU: {result.sar_result.gpu_vram_mb.toFixed(0)} MB VRAM
                  </span>
                )}

                {/* Execution trace — single/change/SAR paths (joint already renders it in JointAnalysisPanel) */}
                {result.intent !== 'joint_analysis' && result.trace && result.trace.length > 0 && (
                  <TraceTimeline steps={result.trace} totalMs={result.elapsed_total_s * 1000} />
                )}
              </div>
            ) : (
              /* Empty state */
              <div className="py-12 text-center">
                <div className="w-16 h-16 rounded-full bg-bg-card border border-border flex items-center justify-center mx-auto mb-4">
                  <BarChart3 className="w-7 h-7 text-text-muted/40" />
                </div>
                <p className="text-sm text-text-muted mb-1">
                  {mode === 'change'
                    ? 'Upload before/after images to detect changes'
                    : mode === 'joint'
                      ? 'Upload optical + SAR images for fused analysis'
                      : 'Upload a satellite image and ask a question'}
                </p>
                <p className="text-xs text-text-muted/60 mb-5">
                  or select a demo scenario above
                </p>
                <div className="inline-block text-left">
                  <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-2">
                    {mode === 'change' ? 'Change Detection' : mode === 'joint' ? 'Joint Analysis' : 'Supported Analysis'}
                  </p>
                  <div className="space-y-1.5 text-xs text-text-secondary">
                    {(mode === 'change'
                      ? ['change']
                      : mode === 'joint'
                        ? ['joint_analysis', 'sar']
                        : ['caption', 'vqa', 'detect', 'classification', 'sar']
                    ).map((intent) => (
                      <div key={intent} className="flex items-center gap-2">
                        <span>{INTENT_ICONS[intent as IntentType]}</span>
                        <span>{INTENT_LABELS[intent as IntentType]}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* History bar */}
      {history.length > 0 && (
        <div className="border-t border-border bg-bg-secondary/50 px-5 py-2">
          <div className="flex items-center gap-2 overflow-x-auto">
            <span className="text-[10px] font-semibold text-text-muted uppercase tracking-wider whitespace-nowrap">
              History
            </span>
            {[...history].reverse().map((entry) => (
              <button
                key={entry.id}
                onClick={() => {
                  setResult(entry.result);
                  setImagePreview(entry.imagePreview);
                  setQuery(entry.query);
                  setMode(entry.mode);
                }}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-border bg-bg-card/50 hover:bg-bg-card-hover hover:border-border-light transition-all whitespace-nowrap text-[11px] text-text-secondary hover:text-text-primary"
              >
                <span style={{ color: INTENT_COLORS[entry.intent] }}>
                  {entry.mode === 'change' ? '🔄' : entry.mode === 'joint' ? '🔗' : INTENT_ICONS[entry.intent]}
                </span>
                <span>{entry.query.slice(0, 30)}{entry.query.length > 30 ? '…' : ''}</span>
                <span className="font-mono text-text-muted">
                  {entry.elapsed_s > 0 ? `${entry.elapsed_s.toFixed(1)}s` : '⚡'}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
