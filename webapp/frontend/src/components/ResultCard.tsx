import { IconAlertTriangle, IconShieldCheck, IconFileText } from "@tabler/icons-react";
import type { DetectionResult } from "../lib/types";

function VerdictBadge({ verdict }: { verdict: DetectionResult["verdict"] }) {
  if (verdict === "malicious") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-ink px-3 py-1 text-[13px] font-semibold text-on-dark">
        <IconAlertTriangle size={14} stroke={2} />
        判定为恶意
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-cream-deeper px-3 py-1 text-[13px] font-semibold text-ink">
      <IconShieldCheck size={14} stroke={2} />
      判定为良性
    </span>
  );
}

export function ResultCard({ result }: { result: DetectionResult }) {
  return (
    <div className="rounded-lg border border-hairline-soft bg-canvas p-8 shadow-[0_4px_12px_rgba(0,0,0,0.04)]">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-medium uppercase tracking-wide text-steel">检测结果</p>
          <p className="mt-1 flex flex-wrap items-center gap-2 font-sans text-base font-medium text-ink">
            <IconFileText size={16} className="shrink-0 text-steel" />
            <span className="min-w-0 break-all">{result.filename}</span>
          </p>
        </div>
        <div className="shrink-0">
          <VerdictBadge verdict={result.verdict} />
        </div>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-6 border-t border-hairline pt-6 sm:grid-cols-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-steel">置信度</p>
          <p className="mt-1 font-display text-4xl leading-tight text-ink">
            {((result.confidence ?? 0) * 100).toFixed(1)}
            <span className="text-xl text-steel">%</span>
          </p>
        </div>
        {result.family && (
          <div className="min-w-0">
            <p className="text-xs font-medium uppercase tracking-wide text-steel">家族分类</p>
            <p className="mt-1 font-display text-2xl leading-tight text-ink break-words">{result.family}</p>
            {result.familyConfidence != null && (
              <p className="mt-1 text-xs text-steel">疑似 · 置信度 {(result.familyConfidence * 100).toFixed(0)}%</p>
            )}
          </div>
        )}
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-steel">双模型一致性</p>
          <p className="mt-2 text-sm font-medium text-ink">
            {result.modelAgreement === "agree" ? "LightGBM 与 MLP 判定一致" : "两模型判定存在分歧"}
          </p>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-2 gap-6 border-t border-hairline pt-6">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-steel">LightGBM 恶意概率</p>
          <p className="mt-1 font-display text-2xl leading-tight text-ink">
            {(result.lgbmScore * 100).toFixed(1)}
            <span className="text-base text-steel">%</span>
          </p>
        </div>
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-steel">MLP 恶意概率</p>
          <p className="mt-1 font-display text-2xl leading-tight text-ink">
            {(result.mlpScore * 100).toFixed(1)}
            <span className="text-base text-steel">%</span>
          </p>
        </div>
      </div>

      {(result.attck?.length ?? 0) > 0 && (
        <div className="mt-6 border-t border-hairline pt-6">
          <p className="text-xs font-medium uppercase tracking-wide text-steel">ATT&amp;CK 战术映射</p>
          <div className="mt-3 flex flex-wrap gap-2">
            {(result.attck ?? []).map((tag) => (
              <span
                key={tag.technique}
                className="rounded-md bg-surface-code px-3 py-1.5 font-mono text-xs text-on-dark"
              >
                <span className="text-on-dark-muted">{tag.tactic}</span>
                {"  ·  "}
                {tag.technique}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="mt-6 border-t border-hairline pt-6">
        <p className="text-xs font-medium uppercase tracking-wide text-steel">LLM 行为分析报告</p>
        <p className="mt-3 whitespace-pre-line text-sm leading-relaxed text-ink-tint">
          {result.llmReport ?? "暂无分析报告"}
        </p>
      </div>
    </div>
  );
}
