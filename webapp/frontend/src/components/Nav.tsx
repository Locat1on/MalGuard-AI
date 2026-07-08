import { NavLink } from "react-router-dom";

const LINKS = [
  { to: "/", label: "检测" },
  { to: "/metrics", label: "评测指标" },
  { to: "/history", label: "历史记录" },
];

export function Nav() {
  return (
    <header className="sticky top-0 z-20 h-16 border-b border-hairline-soft bg-canvas/95 backdrop-blur">
      <div className="mx-auto flex h-full max-w-6xl items-center justify-between px-6">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-on-primary font-display text-lg">
            M
          </div>
          <span className="font-sans text-[15px] font-medium tracking-tight text-ink">
            MalGuard AI
          </span>
        </div>
        <nav className="hidden items-center gap-8 text-sm font-medium md:flex">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              className={({ isActive }) =>
                isActive ? "text-ink" : "text-steel hover:text-ink"
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
        <NavLink
          to="/"
          className="rounded-md bg-ink px-4 py-2 text-sm font-medium text-on-dark"
        >
          开始检测
        </NavLink>
      </div>
    </header>
  );
}
