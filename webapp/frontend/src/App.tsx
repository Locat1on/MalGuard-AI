import { useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Nav } from "./components/Nav";
import { Footer } from "./components/Footer";
import { DetectPage } from "./pages/DetectPage";
import { MetricsPage } from "./pages/MetricsPage";
import { HistoryPage } from "./pages/HistoryPage";
import { analyzeFile, fetchMetrics } from "./lib/api";
import type { DetectionResult, HistoryEntry, ModelMetric } from "./lib/types";

function toHistoryEntry(result: DetectionResult): HistoryEntry {
  return {
    id: crypto.randomUUID(),
    filename: result.filename,
    verdict: result.verdict,
    confidence: result.confidence,
    family: result.family,
    timestamp: new Date().toLocaleString("zh-CN"),
  };
}

function App() {
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);

  useState(() => {
    fetchMetrics().then(setMetrics);
  });

  async function handleAnalyze(file: File) {
    setIsAnalyzing(true);
    setResult(null);
    try {
      const r = await analyzeFile(file);
      setResult(r);
      setHistory((prev) => [toHistoryEntry(r), ...prev]);
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-surface">
        <Nav />
        <Routes>
          <Route
            path="/"
            element={
              <DetectPage result={result} isAnalyzing={isAnalyzing} onAnalyze={handleAnalyze} />
            }
          />
          <Route path="/metrics" element={<MetricsPage metrics={metrics} />} />
          <Route path="/history" element={<HistoryPage entries={history} />} />
        </Routes>
        <Footer />
      </div>
    </BrowserRouter>
  );
}

export default App;
