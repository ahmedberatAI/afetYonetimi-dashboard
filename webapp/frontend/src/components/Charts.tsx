import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CountPoint, PrevalencePoint, TemporalPoint, ProvincePoint } from "../lib/api";
import { formatNumber } from "../lib/api";

const labelColors = ["#b42318", "#047a7a", "#bd7a00", "#2f5f98", "#7f4f24", "#626f47", "#8a3ffc", "#c2410c", "#2563eb"];

type ChartPanelProps = {
  title: string;
  eyebrow: string;
  children: React.ReactNode;
};

export function ChartPanel({ title, eyebrow, children }: ChartPanelProps) {
  return (
    <section className="panel chart-panel">
      <div className="panel-heading">
        <span>{eyebrow}</span>
        <h3>{title}</h3>
      </div>
      {children}
    </section>
  );
}

export function LabelBarChart({ data }: { data: CountPoint[] }) {
  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data.slice().reverse()} layout="vertical" margin={{ top: 4, right: 18, bottom: 4, left: 12 }}>
          <CartesianGrid stroke="#e7edf0" horizontal={false} />
          <XAxis type="number" tickFormatter={(value) => formatNumber(Number(value))} tick={{ fill: "#60707b", fontSize: 12 }} />
          <YAxis type="category" dataKey="name" width={116} tick={{ fill: "#2d3b45", fontSize: 12 }} />
          <Tooltip formatter={(value) => formatNumber(Number(value))} cursor={{ fill: "rgba(180, 35, 24, 0.08)" }} />
          <Bar dataKey="count" radius={[0, 6, 6, 0]} barSize={14}>
            {data.slice().reverse().map((entry, index) => (
              <Cell key={entry.label} fill={labelColors[index % labelColors.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TemporalChart({ data }: { data: TemporalPoint[] }) {
  return (
    <div className="chart-box">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 12, right: 16, bottom: 4, left: 0 }}>
          <defs>
            <linearGradient id="needGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="10%" stopColor="#b42318" stopOpacity={0.42} />
              <stop offset="95%" stopColor="#b42318" stopOpacity={0.04} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#e7edf0" vertical={false} />
          <XAxis dataKey="date" tick={{ fill: "#60707b", fontSize: 12 }} />
          <YAxis tickFormatter={(value) => formatNumber(Number(value))} tick={{ fill: "#60707b", fontSize: 12 }} />
          <Tooltip formatter={(value) => formatNumber(Number(value))} />
          <Area type="monotone" dataKey="rows" stroke="#2f5f98" strokeWidth={2} fill="rgba(47, 95, 152, 0.08)" name="Tweet" />
          <Area type="monotone" dataKey="needs" stroke="#b42318" strokeWidth={2} fill="url(#needGradient)" name="İhtiyaç sinyali" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function PrevalenceTable({ data }: { data: PrevalencePoint[] }) {
  return (
    <div className="prevalence-table" role="table">
      <div className="prevalence-head" role="row">
        <span>Etiket</span>
        <span>Filtre</span>
        <span>Genel</span>
      </div>
      {data.slice(0, 9).map((row) => (
        <div className="prevalence-row" role="row" key={row.label}>
          <span>{row.name}</span>
          <b>{formatNumber(row.filteredRatePct, 2)}%</b>
          <span>{formatNumber(row.fullRatePct, 2)}%</span>
        </div>
      ))}
    </div>
  );
}

export function ProvinceRanking({ data }: { data: ProvincePoint[] }) {
  return (
    <div className="province-list">
      {data.slice(0, 10).map((row, index) => (
        <div className="province-row" key={row.province}>
          <span>{index + 1}</span>
          <strong>{row.province}</strong>
          <b>{formatNumber(row.count)}</b>
        </div>
      ))}
    </div>
  );
}
