import { HeroBand } from "../components/HeroBand";
import { UploadPanel } from "../components/UploadPanel";
import { ResultCard } from "../components/ResultCard";
import type { DetectionResult } from "../lib/types";

interface DetectPageProps {
  result: DetectionResult | null;
  isAnalyzing: boolean;
  error: string | null;
  onAnalyze: (file: File) => void;
}

export function DetectPage({ result, isAnalyzing, error, onAnalyze }: DetectPageProps) {
  return (
    <>
      <HeroBand />
      <section id="upload" className="mx-auto max-w-6xl px-6 py-16">
        <div className="grid gap-6 lg:grid-cols-2">
          <UploadPanel onAnalyze={onAnalyze} isAnalyzing={isAnalyzing} />
          {error ? (
            <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-red-200 bg-red-50 p-8 text-center">
              <p className="text-sm font-medium text-red-700">检测失败</p>
              <p className="text-sm text-red-600">{error}</p>
            </div>
          ) : result ? (
            <ResultCard result={result} />
          ) : (
            <div className="flex items-center justify-center rounded-lg border border-dashed border-hairline-strong bg-canvas p-8 text-sm text-steel">
              检测结果将显示在这里
            </div>
          )}
        </div>
      </section>
    </>
  );
}
