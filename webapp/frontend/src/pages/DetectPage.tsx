import { HeroBand } from "../components/HeroBand";
import { UploadPanel } from "../components/UploadPanel";
import { ResultCard } from "../components/ResultCard";
import type { DetectionResult } from "../lib/types";

interface DetectPageProps {
  result: DetectionResult | null;
  isAnalyzing: boolean;
  onAnalyze: (file: File) => void;
}

export function DetectPage({ result, isAnalyzing, onAnalyze }: DetectPageProps) {
  return (
    <>
      <HeroBand />
      <section id="upload" className="mx-auto max-w-6xl px-6 py-16">
        <div className="grid gap-6 lg:grid-cols-2">
          <UploadPanel onAnalyze={onAnalyze} isAnalyzing={isAnalyzing} />
          {result ? (
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
