import type { ModelMetric } from "../lib/types";

const COLS: { key: keyof ModelMetric; label: string }[] = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1" },
];

export function MetricsSection({ metrics }: { metrics: ModelMetric[] }) {
  return (
    <div className="overflow-hidden rounded-lg border border-hairline-soft bg-canvas">
      <table className="w-full border-collapse text-left text-sm">
        <thead>
          <tr className="border-b border-hairline bg-surface">
            <th className="px-6 py-3 font-medium text-steel">模型</th>
            {COLS.map((c) => (
              <th key={c.key} className="px-6 py-3 font-medium text-steel">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {metrics.map((m, i) => (
            <tr key={m.model} className={i > 0 ? "border-t border-hairline" : ""}>
              <td className="px-6 py-4 font-medium text-ink">{m.model}</td>
              {COLS.map((c) => (
                <td key={c.key} className="px-6 py-4 font-display text-2xl text-ink">
                  {((m[c.key] as number) * 100).toFixed(1)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
