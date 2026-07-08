import { PageHeader } from "../components/PageHeader";
import { HistoryTable } from "../components/HistoryTable";
import type { HistoryEntry } from "../lib/types";

export function HistoryPage({ entries }: { entries: HistoryEntry[] }) {
  return (
    <>
      <PageHeader
        eyebrow="检测记录"
        title="最近检测历史"
        description="本次会话内上传并分析过的样本记录，仅保存在当前浏览器会话中。"
      />
      <section className="mx-auto max-w-6xl px-6 py-12">
        <HistoryTable entries={entries} />
      </section>
    </>
  );
}
