import { PageHeader } from "../components/PageHeader";
import { MetricsSection } from "../components/MetricsSection";
import type { ModelMetric } from "../lib/types";

export function MetricsPage({ metrics }: { metrics: ModelMetric[] }) {
  return (
    <>
      <PageHeader
        eyebrow="模型评测"
        title="深度学习模型 vs 传统基线"
        description="在同一验证集上对比 LightGBM 静态特征基线与本系统的 MLP 深度学习模型，验证深度学习方法带来的精度收益。"
      />
      <section className="mx-auto max-w-6xl px-6 py-12">
        <MetricsSection metrics={metrics} />
      </section>
    </>
  );
}
