import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Database,
  Gauge,
  Layers,
  MapPinned,
  RefreshCw,
  ShieldCheck,
  Table2,
  Zap,
} from "lucide-react";
import { ChartPanel, LabelBarChart, PrevalenceTable, ProvinceRanking, TemporalChart } from "./components/Charts";
import { FilterPanel } from "./components/FilterPanel";
import { HotspotMap } from "./components/HotspotMap";
import { MetricCard } from "./components/MetricCard";
import { TweetTable } from "./components/TweetTable";
import {
  Filters,
  HotspotsResponse,
  OptionsResponse,
  OverviewResponse,
  formatNumber,
  getHotspots,
  getOptions,
  getOverview,
  refreshApi,
} from "./lib/api";

const emptyFilters: Filters = {
  startDate: "",
  endDate: "",
  provinces: [],
  labels: [],
  labelMode: "ANY",
  urgencyMin: null,
  search: "",
};

type View = "operations" | "tweets" | "provenance";

const severityClass = {
  Kritik: "critical",
  Yüksek: "high",
  Orta: "medium",
  İzleme: "watch",
} as const;

function buildDefaultFilters(options: OptionsResponse): Filters {
  return {
    ...emptyFilters,
    startDate: options.dateRange.min ?? "",
    endDate: options.dateRange.max ?? "",
    urgencyMin: options.urgencyRange.min,
  };
}

function App() {
  const [options, setOptions] = useState<OptionsResponse | null>(null);
  const [filters, setFilters] = useState<Filters>(emptyFilters);
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [hotspots, setHotspots] = useState<HotspotsResponse | null>(null);
  const [hour, setHour] = useState<string | null>(null);
  const [hotspotLevel, setHotspotLevel] = useState<"province" | "district" | "neighborhood">("province");
  const [signalMode, setSignalMode] = useState<"count_rows" | "count_any_need" | "sum_urgency">("count_any_need");
  const [playing, setPlaying] = useState(false);
  const [view, setView] = useState<View>("operations");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadOptions = useCallback(async () => {
    const nextOptions = await getOptions();
    setOptions(nextOptions);
    setFilters((current) => {
      if (current.startDate || current.endDate || current.urgencyMin !== null) return current;
      return buildDefaultFilters(nextOptions);
    });
  }, []);

  useEffect(() => {
    setLoading(true);
    loadOptions()
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, [loadOptions]);

  useEffect(() => {
    if (!options) return;
    const controller = new AbortController();
    setLoading(true);
    getOverview(filters)
      .then((data) => {
        if (controller.signal.aborted) return;
        setOverview(data);
        setHour((current) => {
          if (current && data.hours.includes(current)) return current;
          return data.defaultHour ?? data.hours[0] ?? null;
        });
        setError(null);
      })
      .catch((err) => {
        if (!controller.signal.aborted) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [filters, options]);

  useEffect(() => {
    if (!options) return;
    getHotspots(filters, hour, hotspotLevel, signalMode)
      .then((data) => {
        setHotspots(data);
        if (data.hour && data.hour !== hour) setHour(data.hour);
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, [filters, hour, hotspotLevel, signalMode, options]);

  useEffect(() => {
    if (!playing || !hotspots?.hours.length) return;
    const timer = window.setInterval(() => {
      setHour((current) => {
        const hours = hotspots.hours;
        const currentIndex = current ? hours.indexOf(current) : -1;
        const nextIndex = currentIndex >= 0 ? (currentIndex + 1) % hours.length : 0;
        return hours[nextIndex];
      });
    }, 1400);
    return () => window.clearInterval(timer);
  }, [playing, hotspots?.hours]);

  const resetFilters = useCallback(() => {
    if (!options) return;
    setFilters(buildDefaultFilters(options));
    setPlaying(false);
  }, [options]);

  const handleRefresh = useCallback(async () => {
    setLoading(true);
    try {
      await refreshApi();
      await loadOptions();
      const nextOverview = await getOverview(filters);
      setOverview(nextOverview);
      setHour(nextOverview.defaultHour ?? nextOverview.hours[0] ?? null);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [filters, loadOptions]);

  const sourceBadges = useMemo(() => {
    if (!overview?.source && !options?.source) return [];
    const source = overview?.source ?? options!.source;
    return [source.kindLabel, source.experiment, source.generatedAt].filter(Boolean) as string[];
  }, [options, overview]);

  if (!options) {
    return (
      <main className="boot-screen">
        <ShieldCheck size={34} />
        <h1>Afet Yönetimi Local Panel</h1>
        <p>{error ?? "Veri servisi hazırlanıyor..."}</p>
      </main>
    );
  }

  const summary = overview?.summary;

  return (
    <div className="app-shell">
      <FilterPanel options={options} filters={filters} onChange={setFilters} onReset={resetFilters} />

      <main className="workspace">
        <header className="topbar">
          <div className="brand-block">
            <span>Afet Yönetimi</span>
            <h1>Local Operasyon Paneli</h1>
          </div>
          <div className="source-stack">
            {sourceBadges.map((badge) => (
              <span key={badge}>{badge}</span>
            ))}
          </div>
          <button className="primary-action" type="button" onClick={handleRefresh} disabled={loading} title="Veriyi yenile">
            <RefreshCw size={18} className={loading ? "spin" : ""} />
            Yenile
          </button>
        </header>

        {error ? (
          <section className="error-banner">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </section>
        ) : null}

        <section className="metric-grid">
          <MetricCard label="Görünen tweet" value={summary?.filteredRows ?? 0} detail={`${formatNumber(summary?.totalRows ?? 0)} toplam`} tone="blue" icon={Database} />
          <MetricCard label="İhtiyaç sinyali" value={summary?.needSignals ?? 0} detail={`${formatNumber(summary?.needSignalRatePct ?? 0, 1)}% oran`} tone="red" icon={Zap} />
          <MetricCard label="Kapsanan il" value={summary?.provinceCount ?? 0} detail="konumlu kayıt" tone="teal" icon={MapPinned} />
          <MetricCard label="Ort. aciliyet" value={summary?.avgUrgency ?? 0} digits={2} detail={`maks. ${formatNumber(summary?.maxUrgency ?? 0, 1)}`} tone="amber" icon={Gauge} />
          <MetricCard label="En yoğun etiket" value={summary?.topLabel.name ?? "n/a"} detail={`${formatNumber(summary?.topLabel.count ?? 0)} sinyal`} tone="ink" icon={Activity} />
        </section>

        <nav className="view-tabs" aria-label="Görünüm">
          <button className={view === "operations" ? "active" : ""} type="button" onClick={() => setView("operations")}>
            <Layers size={17} /> Operasyon
          </button>
          <button className={view === "tweets" ? "active" : ""} type="button" onClick={() => setView("tweets")}>
            <Table2 size={17} /> Tweetler
          </button>
          <button className={view === "provenance" ? "active" : ""} type="button" onClick={() => setView("provenance")}>
            <ShieldCheck size={17} /> Provenance
          </button>
        </nav>

        {view === "operations" ? (
          <>
            <section className="operations-grid">
              <HotspotMap
                hotspots={hotspots}
                hour={hour}
                playing={playing}
                level={hotspotLevel}
                signalMode={signalMode}
                onHourChange={setHour}
                onPlayingChange={setPlaying}
                onLevelChange={setHotspotLevel}
                onSignalModeChange={setSignalMode}
              />

              <section className="panel priority-panel">
                <div className="panel-heading">
                  <span>Öncelik</span>
                  <h3>Sıcak noktalar</h3>
                </div>
                <div className="hotspot-list">
                  {(hotspots?.points ?? []).slice(0, 12).map((point) => (
                    <article className={`hotspot-row sev-${severityClass[point.severity]}`} key={`${point.rank}-${point.location}`}>
                      <span>{point.rank}</span>
                      <div>
                        <strong>{point.location}</strong>
                        <small>{point.severity} · {formatNumber(point.sharePct, 1)}%</small>
                      </div>
                      <b>{formatNumber(point.signal, 0)}</b>
                    </article>
                  ))}
                </div>
              </section>
            </section>

            <section className="analytics-grid">
              <ChartPanel eyebrow="Etiket" title="Sinyal dağılımı">
                <LabelBarChart data={overview?.labelCounts ?? []} />
              </ChartPanel>
              <ChartPanel eyebrow="Zaman" title="Günlük akış">
                <TemporalChart data={overview?.temporal ?? []} />
              </ChartPanel>
              <ChartPanel eyebrow="Oran" title="Prevalans">
                <PrevalenceTable data={overview?.prevalence ?? []} />
              </ChartPanel>
              <ChartPanel eyebrow="Konum" title="İl sıralaması">
                <ProvinceRanking data={overview?.provinceMap ?? []} />
              </ChartPanel>
            </section>
          </>
        ) : null}

        {view === "tweets" ? <TweetTable tweets={overview?.tweets ?? []} /> : null}

        {view === "provenance" ? (
          <section className="panel provenance-panel">
            <div className="panel-heading">
              <span>Kaynak</span>
              <h3>Veri ve model izi</h3>
            </div>
            <div className="provenance-grid">
              <div>
                <span>Kaynak türü</span>
                <strong>{options.source.kindLabel}</strong>
              </div>
              <div>
                <span>Experiment</span>
                <strong>{options.source.experiment ?? "n/a"}</strong>
              </div>
              <div>
                <span>Eşik</span>
                <strong>{[options.source.thresholdSource, options.source.thresholdType].filter(Boolean).join(" / ") || "n/a"}</strong>
              </div>
              <div>
                <span>Üretim</span>
                <strong>{options.source.generatedAt ?? "n/a"}</strong>
              </div>
              <div className="wide">
                <span>CSV</span>
                <strong>{options.source.path}</strong>
              </div>
              <div className="wide">
                <span>Not</span>
                <strong>{options.metadata.contentOverlapNote ?? options.source.note}</strong>
              </div>
            </div>
          </section>
        ) : null}
      </main>
    </div>
  );
}

export default App;
