export function Footer() {
  return (
    <footer className="mt-8">
      <div
        className="h-3 w-full"
        style={{
          background:
            "linear-gradient(90deg, var(--color-primary), var(--color-sunshine-700), var(--color-sunshine-500), var(--color-yellow-saturated), var(--color-cream))",
        }}
      />
      <div className="bg-cream px-6 py-12 text-ink">
        <div className="mx-auto flex max-w-6xl flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-sm font-medium">MalGuard AI · 恶意软件检测系统</p>
          <p className="text-xs text-steel">
            综合设计课题 · 基于深度学习与 LLM 辅助分析的恶意软件检测 · 仅用于教学与研究演示
          </p>
        </div>
      </div>
    </footer>
  );
}
