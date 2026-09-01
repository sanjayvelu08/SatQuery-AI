import { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, X, Search, Loader2, AlertTriangle, ChevronDown, Image as ImageIcon, MessageSquare, BarChart3 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { AnalyzeResponse, DemoScenario, HistoryEntry, IntentType } from '../types';
import { INTENT_COLORS, INTENT_LABELS, INTENT_ICONS, MODEL_LABELS } from '../lib/constants';
import { formatTiming } from '../lib/utils';
import { getAnnotatedUrl } from '../api/client';
import { IntentBadge } from '../components/IntentBadge';
import { VisualEvidence } from '../components/VisualEvidence';
import { LoadingOverlay } from '../components/LoadingOverlay';
import { DetectionTable } from '../components/DetectionTable';

interface AnalysisWorkspaceProps {
  onBackToLanding: () => void;
  demos: DemoScenario[];
  onAnalyze: (image: File, query: string) => Promise<AnalyzeResponse>;
  onAnalyzeDemo: (demo: DemoScenario) => Promise<AnalyzeResponse>;
}

export function AnalysisWorkspace({
  onBackToLanding,
  demos,
  onAnalyze,
  onAnalyzeDemo,
}: AnalysisWorkspaceProps) {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState('');
  const [query, setQuery] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [selectedDemo, setSelectedDemo] = useState('');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [showDemos, setShowDemos] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Elapsed timer during analysis
  useEffect(() => {
    if (!isAnalyzing) return;
    setElapsedSeconds(0);
    const interval = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const handleFile = useCallback((file: File) => {
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    setSelectedDemo('');
  }, []);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleAnalyze = async () => {
    if (!imageFile || !query.trim()) return;
    setIsAnalyzing(true);
    setError(null);
    try {
      const response = await onAnalyze(imageFile, query.trim());
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
      };
      setHistory((prev) => [...prev, entry].slice(-5));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Demo failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleClear = () => {
    setImageFile(null);
    setImagePreview('');
    setQuery('');
    setResult(null);
    setError(null);
    setSelectedDemo('');
  };

  const annotatedUrl = result ? getAnnotatedUrl(result.annotated_image_url) : null;

  return (
    <div className="flex-1 flex flex-col">
      {/* Demo selector bar */}
      <div className="px-5 py-2.5 border-b border-border bg-bg-secondary/50">
        <div className="flex items-center gap-3">
          <button
            onClick={onBackToLanding}
            className="text-sm text-text-muted hover:text-accent-teal transition-colors"
          >
            ← Home
          </button>
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
                    onClick={() => {
                      setShowDemos(false);
                      handleDemoSelect(demo.name);
                    }}
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
        {/* LEFT COLUMN: Image + Query + Evidence */}
        <div className="lg:w-1/2 flex flex-col border-r border-border overflow-y-auto">
          {/* Image panel */}
          <div className="p-4 border-b border-border">
            <div className="section-label mb-2">
              <ImageIcon className="w-3 h-3" />
              Satellite Image
            </div>
            {imagePreview ? (
              <div className="relative rounded-lg overflow-hidden border border-border bg-bg-primary">
                <img
                  src={imagePreview}
                  alt="Satellite image"
                  className="w-full h-auto max-h-64 object-contain"
                />
                <button
                  onClick={handleClear}
                  className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/50 backdrop-blur flex items-center justify-center text-white/70 hover:text-white hover:bg-black/70 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <label
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                className="flex flex-col items-center gap-2 p-8 border-2 border-dashed border-border rounded-lg bg-bg-primary/30 hover:border-accent-teal/40 hover:bg-accent-teal/3 transition-all cursor-pointer"
              >
                <Upload className="w-6 h-6 text-text-muted" />
                <span className="text-xs text-text-muted">
                  Drag & drop or click to upload
                </span>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileInput}
                  className="hidden"
                />
              </label>
            )}
          </div>

          {/* Query input */}
          <div className="p-4 border-b border-border">
            <div className="section-label mb-2">
              <MessageSquare className="w-3 h-3" />
              Query
            </div>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Describe this image · Are there buildings? · Detect ships in SAR"
              rows={2}
              className="w-full bg-bg-input border border-border rounded-lg px-3 py-2.5 text-sm text-text-primary placeholder:text-text-muted/60 focus:outline-none focus:border-accent-teal focus:ring-1 focus:ring-accent-teal/30 transition-colors resize-none"
            />
            {/* Suggestion chips */}
            {imagePreview && !result && !isAnalyzing && (
              <div className="flex flex-wrap gap-1.5 mt-2">
                {['Describe this image', 'Are there buildings?', 'Classify land use', 'Detect features'].map(
                  (s) => (
                    <button
                      key={s}
                      onClick={() => setQuery(s)}
                      className="px-2.5 py-1 text-[11px] font-medium rounded-full border border-accent-teal/20 bg-accent-teal/5 text-accent-teal hover:bg-accent-teal/15 hover:border-accent-teal/40 transition-all"
                    >
                      {s}
                    </button>
                  ),
                )}
              </div>
            )}
            {/* Action buttons */}
            <div className="flex gap-2 mt-3">
              <button
                onClick={handleAnalyze}
                disabled={!imageFile || !query.trim() || isAnalyzing}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg font-semibold text-sm bg-gradient-to-r from-accent-teal to-accent-teal-hover text-bg-primary shadow-md shadow-accent-teal/20 hover:shadow-lg hover:shadow-accent-teal/30 hover:-translate-y-px active:translate-y-0 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:hover:shadow-md transition-all"
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4" />
                    Analyze
                  </>
                )}
              </button>
              <button
                onClick={handleClear}
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
            />
          </div>
        </div>

        {/* RIGHT COLUMN: Results */}
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

                {/* Answer */}
                {result.supported && result.answer ? (
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
                      {result.answer}
                    </ReactMarkdown>
                  </div>
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
              </div>
            ) : (
              /* Empty state */
              <div className="py-12 text-center">
                <div className="w-16 h-16 rounded-full bg-bg-card border border-border flex items-center justify-center mx-auto mb-4">
                  <BarChart3 className="w-7 h-7 text-text-muted/40" />
                </div>
                <p className="text-sm text-text-muted mb-1">
                  Upload a satellite image and ask a question
                </p>
                <p className="text-xs text-text-muted/60 mb-5">
                  or select a demo scenario above
                </p>
                <div className="inline-block text-left">
                  <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-2">
                    Supported Analysis
                  </p>
                  <div className="space-y-1.5 text-xs text-text-secondary">
                    {(['caption', 'vqa', 'detect', 'classification', 'sar'] as IntentType[]).map((intent) => (
                      <div key={intent} className="flex items-center gap-2">
                        <span>{INTENT_ICONS[intent]}</span>
                        <span>{INTENT_LABELS[intent]}</span>
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
                }}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-border bg-bg-card/50 hover:bg-bg-card-hover hover:border-border-light transition-all whitespace-nowrap text-[11px] text-text-secondary hover:text-text-primary"
              >
                <span style={{ color: INTENT_COLORS[entry.intent] }}>
                  {INTENT_ICONS[entry.intent]}
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
