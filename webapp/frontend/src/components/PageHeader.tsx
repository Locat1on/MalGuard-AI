export function PageHeader({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description?: string;
}) {
  return (
    <div
      className="px-6 pb-10 pt-14"
      style={{
        background:
          "linear-gradient(180deg, var(--color-cream) 0%, var(--color-surface) 100%)",
      }}
    >
      <div className="mx-auto max-w-6xl">
        <p className="text-xs font-semibold uppercase tracking-widest text-primary">{eyebrow}</p>
        <h1 className="mt-2 font-display text-4xl text-ink">{title}</h1>
        {description && <p className="mt-2 max-w-2xl text-sm text-steel">{description}</p>}
      </div>
    </div>
  );
}
