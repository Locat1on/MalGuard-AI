import { Link } from "react-router-dom";

export function HeroBand() {
  return (
    <section
      className="px-6 py-24 text-ink"
      style={{
        background:
          "linear-gradient(135deg, var(--color-sunshine-700) 0%, var(--color-sunshine-900) 60%, var(--color-primary) 100%)",
      }}
    >
      <div className="mx-auto max-w-6xl">
        <h1 className="max-w-2xl font-display text-5xl leading-[1.08] tracking-tight md:text-6xl">
          深度学习驱动的恶意软件检测
        </h1>
        <p className="mt-5 max-w-xl text-lg leading-relaxed text-ink-tint">
          上传 Windows PE 可执行文件，由静态特征深度学习模型给出检测判定，并由大语言模型生成可解释的行为分析报告与
          ATT&amp;CK 战术映射。
        </p>
        <div className="mt-8 flex gap-3">
          <a
            href="#upload"
            className="rounded-md bg-ink px-5 py-2.5 text-sm font-medium text-on-dark"
          >
            立即上传检测
          </a>
          <Link
            to="/metrics"
            className="rounded-md border border-hairline-strong bg-transparent px-5 py-2.5 text-sm font-medium text-ink"
          >
            查看模型评测
          </Link>
        </div>
      </div>
    </section>
  );
}
