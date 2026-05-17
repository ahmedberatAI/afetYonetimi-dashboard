import type { LucideIcon } from "lucide-react";
import { formatNumber } from "../lib/api";

type MetricCardProps = {
  label: string;
  value: number | string;
  detail?: string;
  tone: "red" | "teal" | "amber" | "blue" | "ink";
  icon: LucideIcon;
  digits?: number;
};

export function MetricCard({ label, value, detail, tone, icon: Icon, digits = 0 }: MetricCardProps) {
  const renderedValue = typeof value === "number" ? formatNumber(value, digits) : value;

  return (
    <article className={`metric-card tone-${tone}`}>
      <div className="metric-icon" aria-hidden="true">
        <Icon size={19} strokeWidth={2.1} />
      </div>
      <div>
        <p>{label}</p>
        <strong>{renderedValue}</strong>
        {detail ? <span>{detail}</span> : null}
      </div>
    </article>
  );
}
