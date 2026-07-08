import { useState, useEffect, Component, type ReactNode } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Nav } from "./components/Nav";
import { Footer } from "./components/Footer";
import { DetectPage } from "./pages/DetectPage";
import { MetricsPage } from "./pages/MetricsPage";
import { HistoryPage } from "./pages/HistoryPage";
import { analyzeFile, fetchMetrics } from "./lib/api";
import type { DetectionResult, HistoryEntry, ModelMetric } from "./lib/types";

class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean; error: string }> {
  state = { hasError: false, error: "" };
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-gray-50 p-8">
          <h1 className="text-xl font-bold text-red-600">页面渲染出错</h1>
          <p className="max-w-md text-center text-sm text-gray-600">{this.state.error}</p>
          <button
            onClick={() => this.setState({ hasError: false, error: "" })}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm text-white"
          >
            重试
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function toHistoryEntry(result: DetectionResult): HistoryEntry {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
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
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [metrics, setMetrics] = useState<ModelMetric[]>([]);

  useEffect(() => {
    fetchMetrics().then(setMetrics);
  }, []);

  async function handleAnalyze(file: File) {
    setIsAnalyzing(true);
    setResult(null);
    setError(null);
    try {
      const r = await analyzeFile(file);
      setResult(r);
      setHistory((prev) => [toHistoryEntry(r), ...prev]);
    } catch (err) {
      console.error("Analyze failed:", err);
      setError(err instanceof Error ? err.message : "未知错误");
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <ErrorBoundary>
      <BrowserRouter>
        <div className="min-h-screen bg-surface">
          <Nav />
          <Routes>
            <Route
              path="/"
              element={
                <DetectPage
                  result={result}
                  isAnalyzing={isAnalyzing}
                  error={error}
                  onAnalyze={handleAnalyze}
                />
              }
            />
            <Route path="/metrics" element={<MetricsPage metrics={metrics} />} />
            <Route path="/history" element={<HistoryPage entries={history} />} />
          </Routes>
          <Footer />
        </div>
      </BrowserRouter>
    </ErrorBoundary>
  );
}

export default App;
