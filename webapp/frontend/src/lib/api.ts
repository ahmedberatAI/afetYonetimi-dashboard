const API_BASE =
  import.meta.env.VITE_API_BASE ||
  (window.location.port === "5173" ? "http://127.0.0.1:8787" : "");

export type LabelOption = {
  id: string;
  name: string;
};

export type SourceInfo = {
  label: string;
  kind: string;
  kindLabel: string;
  note: string;
  path: string;
  metaPath: string | null;
  generatedAt: string | null;
  experiment: string | null;
  thresholdSource: string | null;
  thresholdType: string | null;
};

export type OptionsResponse = {
  source: SourceInfo;
  dateRange: { min: string | null; max: string | null };
  urgencyRange: { min: number; max: number };
  labels: LabelOption[];
  provinces: string[];
  districts: string[];
  rowCount: number;
  metadata: {
    rowCount?: number;
    rowsBefore?: number;
    rowsAfter?: number;
    duplicateRowsRemoved?: number;
    canonical: boolean;
    contentOverlapNote?: string | null;
  };
};

export type Filters = {
  startDate: string;
  endDate: string;
  provinces: string[];
  labels: string[];
  labelMode: "ANY" | "ALL";
  urgencyMin: number | null;
  search: string;
};

export type MetricSummary = {
  totalRows: number;
  filteredRows: number;
  needSignals: number;
  needSignalRatePct: number;
  provinceCount: number;
  avgUrgency: number;
  maxUrgency: number;
  topLabel: { label: string | null; name: string; count: number };
};

export type CountPoint = {
  label: string;
  name: string;
  count: number;
};

export type PrevalencePoint = {
  label: string;
  name: string;
  fullPositive: number;
  fullRatePct: number;
  filteredPositive: number;
  filteredRatePct: number;
};

export type TemporalPoint = {
  date: string;
  rows: number;
  needs: number;
  urgency: number;
};

export type ProvincePoint = {
  province: string;
  count: number;
  lat: number;
  lon: number;
};

export type TweetRow = {
  id: string;
  date: string;
  time: string;
  province: string;
  district: string;
  neighborhood: string;
  text: string;
  urgency: number;
  labels: LabelOption[];
};

export type OverviewResponse = {
  summary: MetricSummary;
  labelCounts: CountPoint[];
  prevalence: PrevalencePoint[];
  temporal: TemporalPoint[];
  provinceMap: ProvincePoint[];
  tweets: TweetRow[];
  hours: string[];
  defaultHour: string | null;
  source: SourceInfo;
};

export type HotspotPoint = {
  rank: number;
  location: string;
  province: string;
  district: string;
  neighborhood: string;
  lat: number;
  lon: number;
  signal: number;
  sharePct: number;
  severity: "Kritik" | "Yüksek" | "Orta" | "İzleme";
};

export type HotspotsResponse = {
  hour: string | null;
  hours: string[];
  points: HotspotPoint[];
  stats: {
    totalSignal: number;
    hotspots: number;
    topSignal: number;
    geoCoveragePct: number;
  };
};

export function formatNumber(value: number | null | undefined, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return "0";
  return new Intl.NumberFormat("tr-TR", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

export function formatHour(value: string | null | undefined) {
  if (!value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("tr-TR", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function paramsFromFilters(filters: Filters) {
  const params = new URLSearchParams();
  if (filters.startDate) params.set("start_date", filters.startDate);
  if (filters.endDate) params.set("end_date", filters.endDate);
  if (filters.provinces.length) params.set("provinces", filters.provinces.join(","));
  if (filters.labels.length) params.set("labels", filters.labels.join(","));
  params.set("label_mode", filters.labelMode);
  if (filters.urgencyMin !== null) params.set("urgency_min", String(filters.urgencyMin));
  if (filters.search.trim()) params.set("search", filters.search.trim());
  return params;
}

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json() as Promise<T>;
}

export function getOptions() {
  return request<OptionsResponse>("/api/options");
}

export function getOverview(filters: Filters) {
  const params = paramsFromFilters(filters);
  return request<OverviewResponse>(`/api/overview?${params.toString()}`);
}

export function getHotspots(
  filters: Filters,
  hour: string | null,
  level: "province" | "district" | "neighborhood",
  signalMode: "count_rows" | "count_any_need" | "sum_urgency",
) {
  const params = paramsFromFilters(filters);
  if (hour) params.set("hour", hour);
  params.set("level", level);
  params.set("signal_mode", signalMode);
  return request<HotspotsResponse>(`/api/hotspots?${params.toString()}`);
}

export async function refreshApi() {
  const response = await fetch(`${API_BASE}/api/refresh`, { method: "POST" });
  if (!response.ok) throw new Error(await response.text());
  return response.json() as Promise<{ ok: boolean; rows: number; source: string }>;
}
