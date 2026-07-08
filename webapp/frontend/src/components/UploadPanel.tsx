import { useRef, useState } from "react";
import { IconUpload, IconLoader2 } from "@tabler/icons-react";

interface UploadPanelProps {
  onAnalyze: (file: File) => void;
  isAnalyzing: boolean;
}

export function UploadPanel({ onAnalyze, isAnalyzing }: UploadPanelProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (file) onAnalyze(file);
  }

  return (
    <div className="rounded-lg border border-hairline-soft bg-canvas p-8">
      <h2 className="font-sans text-lg font-medium text-ink">上传待检测文件</h2>
      <p className="mt-1 text-sm text-steel">
        支持 Windows PE 可执行文件（.exe / .dll）。文件仅用于本次分析，不会被保留。
      </p>

      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        disabled={isAnalyzing}
        className={`mt-6 flex w-full flex-col items-center justify-center gap-3 rounded-md border-2 border-dashed px-6 py-14 text-center transition-colors ${
          isDragging
            ? "border-primary bg-cream-light"
            : "border-hairline-strong bg-surface"
        }`}
      >
        {isAnalyzing ? (
          <>
            <IconLoader2 className="animate-spin text-primary" size={32} stroke={1.5} />
            <span className="text-sm font-medium text-steel">正在分析样本，请稍候...</span>
          </>
        ) : (
          <>
            <IconUpload className="text-steel" size={32} stroke={1.5} />
            <span className="text-sm font-medium text-ink">
              拖拽文件到此处，或点击选择
            </span>
          </>
        )}
      </button>

      <input
        ref={inputRef}
        type="file"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
