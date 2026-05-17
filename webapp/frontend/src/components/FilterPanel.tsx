import { RotateCcw, Search, SlidersHorizontal } from "lucide-react";
import type { Filters, OptionsResponse } from "../lib/api";

type FilterPanelProps = {
  options: OptionsResponse;
  filters: Filters;
  onChange: (filters: Filters) => void;
  onReset: () => void;
};

function toggleValue(values: string[], value: string) {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

export function FilterPanel({ options, filters, onChange, onReset }: FilterPanelProps) {
  const urgencyMax = Math.max(options.urgencyRange.max, options.urgencyRange.min);
  const urgencyValue = filters.urgencyMin ?? options.urgencyRange.min;

  return (
    <aside className="filter-panel">
      <div className="filter-header">
        <div>
          <span>Kontrol</span>
          <h2>Filtreler</h2>
        </div>
        <button className="icon-button" type="button" onClick={onReset} title="Filtreleri sıfırla" aria-label="Filtreleri sıfırla">
          <RotateCcw size={18} />
        </button>
      </div>

      <div className="filter-group">
        <label>Tarih aralığı</label>
        <div className="date-grid">
          <input
            type="date"
            min={options.dateRange.min ?? undefined}
            max={options.dateRange.max ?? undefined}
            value={filters.startDate}
            onChange={(event) => onChange({ ...filters, startDate: event.target.value })}
          />
          <input
            type="date"
            min={options.dateRange.min ?? undefined}
            max={options.dateRange.max ?? undefined}
            value={filters.endDate}
            onChange={(event) => onChange({ ...filters, endDate: event.target.value })}
          />
        </div>
      </div>

      <div className="filter-group">
        <label>İl</label>
        <div className="check-list">
          {options.provinces.map((province) => (
            <label className="check-row" key={province}>
              <input
                type="checkbox"
                checked={filters.provinces.includes(province)}
                onChange={() => onChange({ ...filters, provinces: toggleValue(filters.provinces, province) })}
              />
              <span>{province}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="filter-group">
        <label>İhtiyaç etiketi</label>
        <div className="label-grid">
          {options.labels.map((label) => (
            <button
              className={filters.labels.includes(label.id) ? "label-chip active" : "label-chip"}
              key={label.id}
              type="button"
              onClick={() => onChange({ ...filters, labels: toggleValue(filters.labels, label.id) })}
            >
              {label.name}
            </button>
          ))}
        </div>
        <div className="segmented compact" aria-label="Etiket filtresi modu">
          <button
            type="button"
            className={filters.labelMode === "ANY" ? "active" : ""}
            onClick={() => onChange({ ...filters, labelMode: "ANY" })}
          >
            ANY
          </button>
          <button
            type="button"
            className={filters.labelMode === "ALL" ? "active" : ""}
            onClick={() => onChange({ ...filters, labelMode: "ALL" })}
          >
            ALL
          </button>
        </div>
      </div>

      <div className="filter-group">
        <label className="range-label">
          <span>Min. aciliyet</span>
          <b>{urgencyValue}</b>
        </label>
        <div className="range-row">
          <SlidersHorizontal size={17} aria-hidden="true" />
          <input
            type="range"
            min={options.urgencyRange.min}
            max={urgencyMax}
            step={1}
            value={urgencyValue}
            onChange={(event) => onChange({ ...filters, urgencyMin: Number(event.target.value) })}
          />
        </div>
      </div>

      <div className="filter-group">
        <label>Metin ara</label>
        <div className="search-box">
          <Search size={17} aria-hidden="true" />
          <input
            type="search"
            value={filters.search}
            placeholder="tweet, konum, ihtiyaç..."
            onChange={(event) => onChange({ ...filters, search: event.target.value })}
          />
        </div>
      </div>
    </aside>
  );
}
