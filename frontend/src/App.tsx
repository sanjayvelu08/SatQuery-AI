import { useState, useEffect, useCallback } from 'react';
import type { AnalyzeResponse, DemoScenario } from './types';
import { Header } from './components/Header';
import { AboutModal } from './components/AboutModal';
import { LandingScreen } from './screens/LandingScreen';
import { AnalysisWorkspace } from './screens/AnalysisWorkspace';
import { fetchDemos, analyzeImage, analyzeDemo } from './api/client';

type Screen = 'landing' | 'workspace';

export default function App() {
  const [screen, setScreen] = useState<Screen>('landing');
  const [showAbout, setShowAbout] = useState(false);
  const [demos, setDemos] = useState<DemoScenario[]>([]);
  

  // Fetch demos on mount
  useEffect(() => {
    fetchDemos()
      .then(setDemos)
      .catch(() => {
        // Fallback: hardcoded demo data from Python backend
        console.warn('Could not fetch demos from API, using built-in fallback data');
      });
  }, []);

  const handleUpload = useCallback((_file: File) => {
    setScreen('workspace');
  }, []);

  const handleAnalyze = useCallback(async (image: File, query: string): Promise<AnalyzeResponse> => {
    return analyzeImage(image, query);
  }, []);

  const handleAnalyzeDemo = useCallback(async (demo: DemoScenario): Promise<AnalyzeResponse> => {
    return analyzeDemo(demo.name);
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-bg-primary">
      <Header onAbout={() => setShowAbout(true)} demoCount={demos.length} />

      {screen === 'landing' ? (
        <LandingScreen onUpload={handleUpload} />
      ) : (
        <AnalysisWorkspace
          onBackToLanding={() => setScreen('landing')}
          demos={demos}
          onAnalyze={handleAnalyze}
          onAnalyzeDemo={handleAnalyzeDemo}
        />
      )}

      <AboutModal isOpen={showAbout} onClose={() => setShowAbout(false)} />
    </div>
  );
}
