import { ChevronLeft, ChevronRight, Pause, Play, Radio } from "lucide-react";
import type { HotspotsResponse } from "../lib/api";
import { formatHour, formatNumber } from "../lib/api";

type HotspotMapProps = {
  hotspots: HotspotsResponse | null;
  hour: string | null;
  playing: boolean;
  level: "province" | "district" | "neighborhood";
  signalMode: "count_rows" | "count_any_need" | "sum_urgency";
  onHourChange: (hour: string | null) => void;
  onPlayingChange: (playing: boolean) => void;
  onLevelChange: (level: "province" | "district" | "neighborhood") => void;
  onSignalModeChange: (mode: "count_rows" | "count_any_need" | "sum_urgency") => void;
};

const severityColor = {
  Kritik: "#8f1d14",
  Yüksek: "#c2410c",
  Orta: "#bd7a00",
  İzleme: "#047a7a",
};

const LON_MIN = 34.2;
const LON_MAX = 41.6;
const LAT_MIN = 35.6;
const LAT_MAX = 38.9;

function project(lon: number, lat: number) {
  const x = ((lon - LON_MIN) / (LON_MAX - LON_MIN)) * 1000;
  const y = ((LAT_MAX - lat) / (LAT_MAX - LAT_MIN)) * 520 + 20;
  return {
    x: Math.max(24, Math.min(976, x)),
    y: Math.max(28, Math.min(532, y)),
  };
}

export function HotspotMap({
  hotspots,
  hour,
  playing,
  level,
  signalMode,
  onHourChange,
  onPlayingChange,
  onLevelChange,
  onSignalModeChange,
}: HotspotMapProps) {
  const hours = hotspots?.hours ?? [];
  const activeHour = hour ?? hotspots?.hour ?? null;
  const activeIndex = activeHour ? Math.max(0, hours.indexOf(activeHour)) : 0;
  const topSignal = Math.max(1, hotspots?.stats.topSignal ?? 1);

  const moveHour = (direction: -1 | 1) => {
    if (!hours.length) return;
    const nextIndex = Math.max(0, Math.min(hours.length - 1, activeIndex + direction));
    onHourChange(hours[nextIndex]);
    onPlayingChange(false);
  };

  return (
    <section className="panel map-panel">
      <div className="panel-heading map-heading">
        <div>
          <span>Canlı Harita</span>
          <h3>Saatlik yardım sinyalleri</h3>
        </div>
        <div className="map-stat-strip">
          <span>{formatHour(activeHour)}</span>
          <b>{formatNumber(hotspots?.stats.totalSignal ?? 0, 0)} sinyal</b>
        </div>
      </div>

      <div className="map-controls">
        <div className="segmented" aria-label="Konum seviyesi">
          {(["province", "district", "neighborhood"] as const).map((item) => (
            <button className={level === item ? "active" : ""} key={item} type="button" onClick={() => onLevelChange(item)}>
              {item === "province" ? "İl" : item === "district" ? "İlçe" : "Mahalle"}
            </button>
          ))}
        </div>
        <div className="segmented" aria-label="Sinyal modu">
          <button className={signalMode === "count_any_need" ? "active" : ""} type="button" onClick={() => onSignalModeChange("count_any_need")}>
            İhtiyaç
          </button>
          <button className={signalMode === "count_rows" ? "active" : ""} type="button" onClick={() => onSignalModeChange("count_rows")}>
            Tweet
          </button>
          <button className={signalMode === "sum_urgency" ? "active" : ""} type="button" onClick={() => onSignalModeChange("sum_urgency")}>
            Aciliyet
          </button>
        </div>
      </div>

      <div className="timeline-row">
        <button className="icon-button" type="button" title="Önceki saat" aria-label="Önceki saat" onClick={() => moveHour(-1)} disabled={!hours.length}>
          <ChevronLeft size={18} />
        </button>
        <button
          className="icon-button play-button"
          type="button"
          title={playing ? "Duraklat" : "Oynat"}
          aria-label={playing ? "Duraklat" : "Oynat"}
          onClick={() => onPlayingChange(!playing)}
          disabled={hours.length < 2}
        >
          {playing ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(0, hours.length - 1)}
          step={1}
          value={activeIndex}
          onChange={(event) => onHourChange(hours[Number(event.target.value)] ?? null)}
          disabled={!hours.length}
        />
        <button className="icon-button" type="button" title="Sonraki saat" aria-label="Sonraki saat" onClick={() => moveHour(1)} disabled={!hours.length}>
          <ChevronRight size={18} />
        </button>
      </div>

      <div className="map-surface">
        <svg viewBox="0 0 1000 560" role="img" aria-label="Afet bölgesi sinyal haritası">
          <defs>
            <pattern id="mapGrid" width="50" height="50" patternUnits="userSpaceOnUse">
              <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#d9e3e7" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="1000" height="560" rx="8" fill="#eef4f2" />
          <rect width="1000" height="560" rx="8" fill="url(#mapGrid)" opacity="0.74" />
          <path
            d="M58 353 C126 261 184 247 259 270 C330 292 368 197 450 203 C524 209 560 143 637 169 C706 193 770 179 844 232 C901 273 940 342 934 408 C828 374 760 421 679 396 C614 377 562 430 496 401 C420 368 360 413 292 385 C206 349 148 414 58 353 Z"
            fill="#d7e8e4"
            stroke="#9fb9b2"
            strokeWidth="2"
            opacity="0.86"
          />
          {(hotspots?.points ?? []).slice(0, 160).map((point) => {
            const { x, y } = project(point.lon, point.lat);
            const radius = Math.max(7, Math.min(42, Math.sqrt(point.signal / topSignal) * 40));
            const color = severityColor[point.severity];
            return (
              <g key={`${point.rank}-${point.location}`}>
                <circle cx={x} cy={y} r={radius + 7} fill={color} opacity="0.1" />
                <circle cx={x} cy={y} r={radius} fill={color} opacity="0.82" stroke="#fff" strokeWidth="2">
                  <title>
                    {point.location} | {formatNumber(point.signal, 0)} sinyal | {point.severity}
                  </title>
                </circle>
              </g>
            );
          })}
          {(hotspots?.points ?? []).slice(0, 6).map((point) => {
            const { x, y } = project(point.lon, point.lat);
            return (
              <text key={`label-${point.rank}`} x={Math.min(880, x + 16)} y={Math.max(24, y - 12)} className="map-label">
                {point.rank}. {point.province || point.location}
              </text>
            );
          })}
        </svg>
      </div>

      <div className="map-footer">
        <span>
          <Radio size={15} /> {formatNumber(hotspots?.stats.hotspots ?? 0)} nokta
        </span>
        <span>Koordinat kapsamı {formatNumber(hotspots?.stats.geoCoveragePct ?? 0, 1)}%</span>
      </div>
    </section>
  );
}
