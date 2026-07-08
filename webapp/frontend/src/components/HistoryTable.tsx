import type { HistoryEntry } from "../lib/types";

export function HistoryTable({ entries }: { entries: HistoryEntry[] }) {
  return (
    <div className="rounded-lg border border-hairline-soft bg-canvas">
      {entries.length === 0 ? (
        <p className="px-6 py-10 text-center text-sm text-steel">
          暂无检测记录，前往检测页上传文件开始第一次检测。
        </p>
      ) : (
        entries.map((e, i) => (
          <div
            key={e.id}
            className={`flex items-center justify-between px-6 py-4 ${i > 0 ? "border-t border-hairline" : ""}`}
          >
            <div>
              <p className="text-sm font-medium text-ink">{e.filename}</p>
              <p className="mt-0.5 text-xs text-steel">{e.timestamp}</p>
            </div>
            <div className="flex items-center gap-4">
              {e.family && (
                <span className="rounded-full bg-cream-deeper px-3 py-1 text-xs font-semibold text-ink">
                  {e.family}
                </span>
              )}
              <span
                className={`rounded-full px-3 py-1 text-xs font-semibold ${
                  e.verdict === "malicious" ? "bg-ink text-on-dark" : "bg-cream-deeper text-ink"
                }`}
              >
                {e.verdict === "malicious" ? "恶意" : "良性"}
              </span>
              <span className="w-14 text-right font-display text-lg text-ink">
                {(e.confidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        ))
      )}
    </div>
  );
}
