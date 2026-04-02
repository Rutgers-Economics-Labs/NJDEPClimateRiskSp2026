import { useEffect, useId, useMemo, useState } from 'react'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000'

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '--'
  }
  return Number(value).toFixed(digits)
}

function formatInteger(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '--'
  }
  return Number(value).toLocaleString()
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '--'
  }
  return `${Number(value).toFixed(digits)}%`
}

function rollingAverage(values, windowSize = 4) {
  return values.map((_, index) => {
    const slice = values
      .slice(Math.max(0, index - windowSize + 1), index + 1)
      .filter((value) => value !== null && value !== undefined)
    if (!slice.length) {
      return null
    }
    return slice.reduce((sum, value) => sum + value, 0) / slice.length
  })
}

async function fetchJson(url) {
  const response = await fetch(url)
  const payload = await response.json()
  if (!response.ok) {
    throw new Error(payload.detail || 'The backend request failed.')
  }
  return payload
}

function TimeSeriesChart({ series, metricView }) {
  const chartId = useId()
  const width = 920
  const height = 300
  const padding = { top: 24, right: 20, bottom: 42, left: 56 }

  const points = useMemo(() => {
    if (!series.length) {
      return null
    }

    const rawValues =
      metricView === 'aaa' ? series.map((point) => point.avg_synthetic_aaa) : series.map((point) => point.avg_spread_bps)
    const smoothValues =
      metricView === 'aaa' ? rollingAverage(rawValues, 4) : series.map((point) => point.spread_bps_rolling_4w)
    const allValues = [...rawValues, ...smoothValues].filter((value) => value !== null && value !== undefined)
    const minValue = Math.min(...allValues)
    const maxValue = Math.max(...allValues)
    const range = Math.max(maxValue - minValue, 1)
    const innerWidth = width - padding.left - padding.right
    const innerHeight = height - padding.top - padding.bottom

    const toPoint = (value, index) => {
      const x = padding.left + (series.length === 1 ? innerWidth / 2 : (index / (series.length - 1)) * innerWidth)
      const y = padding.top + innerHeight - ((value - minValue) / range) * innerHeight
      return `${x},${y}`
    }

    return {
      rawPath: rawValues.map(toPoint).join(' '),
      smoothPath: smoothValues.map(toPoint).join(' '),
      minValue,
      maxValue,
      innerHeight,
    }
  }, [metricView, series])

  if (!series.length || !points) {
    return (
      <div className="chart-empty">
        <p>No time-series observations were available for this selection.</p>
      </div>
    )
  }

  const ticks = [points.minValue, (points.minValue + points.maxValue) / 2, points.maxValue]
  const tickLabels = [series[0], series[Math.floor(series.length / 2)], series[series.length - 1]]
  const title =
    metricView === 'aaa' ? 'Synthetic AAA proxy through time' : 'Premium over synthetic AAA through time'
  const rawLabel = metricView === 'aaa' ? 'Weekly synthetic AAA' : 'Weekly average spread'
  const smoothLabel = metricView === 'aaa' ? '4-period rolling AAA' : '4-period rolling premium'

  return (
    <div className="chart-shell">
      <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg" role="img" aria-labelledby={chartId}>
        <title id={chartId}>{title}</title>

        {ticks.map((tickValue) => {
          const y =
            padding.top +
            points.innerHeight -
            ((tickValue - points.minValue) / Math.max(points.maxValue - points.minValue, 1)) * points.innerHeight
          return (
            <g key={tickValue}>
              <line x1={padding.left} x2={width - padding.right} y1={y} y2={y} className="chart-grid" />
              <text x={padding.left - 10} y={y + 4} className="chart-axis-label chart-axis-left">
                {formatNumber(tickValue, 1)}
              </text>
            </g>
          )
        })}

        <polyline fill="none" stroke="rgba(44, 107, 79, 0.32)" strokeWidth="2" points={points.rawPath} />
        <polyline fill="none" stroke="var(--accent)" strokeWidth="3.5" points={points.smoothPath} />

        {tickLabels.map((point, index) => {
          const x = padding.left + (index / (tickLabels.length - 1 || 1)) * (width - padding.left - padding.right)
          return (
            <text key={`${point.date_bucket}-${index}`} x={x} y={height - 10} className="chart-axis-label chart-axis-bottom">
              {point.date_bucket}
            </text>
          )
        })}
      </svg>

      <div className="chart-legend">
        <span className="legend-item">
          <span className="legend-swatch legend-swatch-raw" />
          {rawLabel}
        </span>
        <span className="legend-item">
          <span className="legend-swatch legend-swatch-smooth" />
          {smoothLabel}
        </span>
      </div>
    </div>
  )
}

function App() {
  const [options, setOptions] = useState(null)
  const [level, setLevel] = useState('municipality')
  const [view, setView] = useState('weekly')
  const [metricView, setMetricView] = useState('premium')
  const [selectedMunicipality, setSelectedMunicipality] = useState('')
  const [selectedCounty, setSelectedCounty] = useState('')
  const [summary, setSummary] = useState(null)
  const [series, setSeries] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchJson(`${API_BASE}/api/options`)
      .then((data) => {
        setOptions(data)
        setSelectedMunicipality(data.default_municipality || data.municipalities[0] || '')
        setSelectedCounty(data.default_county || data.counties[0] || '')
      })
      .catch(() => setError("Couldn't load dashboard options from the backend."))
  }, [])

  const activeName =
    level === 'municipality' ? selectedMunicipality : level === 'county' ? selectedCounty : null

  useEffect(() => {
    if (!options) {
      return
    }

    if (level === 'municipality' && !selectedMunicipality) {
      return
    }

    if (level === 'county' && !selectedCounty) {
      return
    }

    setLoading(true)
    setError('')

    const params = new URLSearchParams({ level, view })
    const tsParams = new URLSearchParams({ level, view, metric: 'spread_bps' })
    if (activeName) {
      params.set('name', activeName)
      tsParams.set('name', activeName)
    }

    Promise.all([
      fetchJson(`${API_BASE}/api/summary?${params.toString()}`),
      fetchJson(`${API_BASE}/api/timeseries?${tsParams.toString()}`),
    ])
      .then(([summaryData, timeseriesData]) => {
        setSummary(summaryData)
        setSeries(timeseriesData.series || [])
      })
      .catch((fetchError) => setError(fetchError.message || 'The premium explorer could not load.'))
      .finally(() => setLoading(false))
  }, [activeName, level, options, selectedCounty, selectedMunicipality, view])

  return (
    <div className="dashboard-container">
      <header className="hero-panel">
        <div>
          <div className="hero-brand">
            <img src="/njdep_logo.png" alt="New Jersey Department of Environmental Protection logo" className="hero-logo" />
            <div>
              <p className="eyebrow">Rutgers Economics Labs x NJDEP</p>
              <p className="brand-caption">Climate Change & Resilience Premiums</p>
            </div>
          </div>
          <h1>NJ Municipal Resilience Premiums</h1>
          <p className="subtitle">
            Choose a municipality, county, or the whole state to view municipal bond premiums over the AAA benchmark. (currently using synthetic AAA data)
          </p>
        </div>
        <div className="hero-meta">
          <span>Date range</span>
          <strong>
            {options ? `${options.date_min} to ${options.date_max}` : 'Loading range...'}
          </strong>
        </div>
      </header>

      <main className="dashboard-main">
        <section className="card controls-card">
          <div className="toggle-row">
            <div className="segmented-control">
              {['municipality', 'county', 'state'].map((option) => (
                <button
                  key={option}
                  type="button"
                  className={level === option ? 'segment active' : 'segment'}
                  onClick={() => setLevel(option)}
                >
                  {option[0].toUpperCase() + option.slice(1)}
                </button>
              ))}
            </div>

            <div className="segmented-control">
              {['weekly', 'monthly'].map((option) => (
                <button
                  key={option}
                  type="button"
                  className={view === option ? 'segment active' : 'segment'}
                  onClick={() => setView(option)}
                >
                  {option[0].toUpperCase() + option.slice(1)}
                </button>
              ))}
            </div>
          </div>

          <div className="selector-grid">
            <label className="field">
              <span>Municipality search</span>
              <input
                list="municipalities"
                value={selectedMunicipality}
                onChange={(event) => setSelectedMunicipality(event.target.value.toUpperCase())}
                disabled={!options || level !== 'municipality'}
                placeholder="Start typing a municipality"
              />
              <datalist id="municipalities">
                {(options?.municipalities || []).map((municipality) => (
                  <option key={municipality} value={municipality} />
                ))}
              </datalist>
            </label>

            <label className="field">
              <span>County</span>
              <select
                value={selectedCounty}
                onChange={(event) => setSelectedCounty(event.target.value)}
                disabled={!options || level !== 'county'}
              >
                {(options?.counties || []).map((county) => (
                  <option key={county} value={county}>
                    {county}
                  </option>
                ))}
              </select>
            </label>

            <div className="field field-static">
              <span>Current focus</span>
              <strong>
                {level === 'state' ? 'New Jersey' : activeName || 'Waiting for selection'}
              </strong>
            </div>
          </div>
        </section>

        {error ? (
          <section className="card error-card">
            <h2>Backend connection error</h2>
            <p>{error}</p>
          </section>
        ) : null}

        <section className="stats-grid">
          <article className="card stat-card">
            <span className="stat-label">Latest premium</span>
            <strong>{formatNumber(summary?.latest_spread_bps_rolling_4w)} bps</strong>
            <p>Latest 4-period rolling premium over the synthetic AAA proxy.</p>
          </article>
          <article className="card stat-card">
            <span className="stat-label">Average premium</span>
            <strong>{formatNumber(summary?.avg_spread_bps)} bps</strong>
            <p>Average spread across the selected weekly or monthly series.</p>
          </article>
          <article className="card stat-card">
            <span className="stat-label">Peak premium</span>
            <strong>{formatNumber(summary?.peak_spread_bps)} bps</strong>
            <p>Highest observed average premium in the displayed series.</p>
          </article>
          <article className="card stat-card">
            <span className="stat-label">Trade coverage</span>
            <strong>{formatInteger(summary?.trade_count)}</strong>
            <p>{formatInteger(summary?.cusip_count)} unique CUSIPs across this selection.</p>
          </article>
          <article className="card stat-card">
            <span className="stat-label">Resilient share</span>
            <strong>{formatPercent(summary?.resilient_share_pct)}</strong>
            <p>
              {summary?.level === 'municipality'
                ? 'Municipality-level resilience flag from the CRS-based classification.'
                : `${formatInteger(summary?.resilient_muni_count)} resilient municipalities across ${formatInteger(summary?.municipality_count)} municipalities in scope.`}
            </p>
          </article>
        </section>

        <section className="card chart-card">
          <div className="section-header">
            <div>
              <h2>{metricView === 'aaa' ? 'Synthetic AAA proxy through time' : 'Premium over synthetic AAA through time'}</h2>
              <p>
                {loading
                  ? 'Refreshing time series...'
                  : metricView === 'aaa'
                    ? `Showing the benchmark path used for ${level === 'state' ? 'New Jersey' : activeName}.`
                    : `Showing ${view} observations for ${level === 'state' ? 'New Jersey' : activeName}.`}
              </p>
            </div>
            <div className="segmented-control">
              {[
                ['premium', 'Premium'],
                ['aaa', 'AAA Proxy'],
              ].map(([option, label]) => (
                <button
                  key={option}
                  type="button"
                  className={metricView === option ? 'segment active' : 'segment'}
                  onClick={() => setMetricView(option)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <TimeSeriesChart series={series} metricView={metricView} />
        </section>

        <section className="metadata-grid">
          <article className="card detail-card">
            <h2>Selection details</h2>
            <dl className="detail-list">
              <div>
                <dt>Level</dt>
                <dd>{summary?.level || '--'}</dd>
              </div>
              <div>
                <dt>Name</dt>
                <dd>{summary?.label || summary?.name || '--'}</dd>
              </div>
              <div>
                <dt>County</dt>
                <dd>{summary?.county || '--'}</dd>
              </div>
              <div>
                <dt>Resilient</dt>
                <dd>{summary?.is_resilient === null || summary?.is_resilient === undefined ? '--' : summary.is_resilient ? 'Yes' : 'No'}</dd>
              </div>
              <div>
                <dt>CRS class</dt>
                <dd>{summary?.crs_class ?? '--'}</dd>
              </div>
              <div>
                <dt>CRS discount</dt>
                <dd>{formatPercent(summary?.crs_discount_pct)}</dd>
              </div>
              <div>
                <dt>Median income</dt>
                <dd>{summary?.median_household_income ? `$${formatInteger(summary.median_household_income)}` : '--'}</dd>
              </div>
            </dl>
          </article>

          <article className="card detail-card">
            <h2>Flood exposure snapshot</h2>
            <dl className="detail-list">
              <div>
                <dt>2ft exposure</dt>
                <dd>{summary?.flooded_pct_2ft !== undefined && summary?.flooded_pct_2ft !== null ? `${formatNumber(summary.flooded_pct_2ft)}%` : '--'}</dd>
              </div>
              <div>
                <dt>5ft exposure</dt>
                <dd>{summary?.flooded_pct_5ft !== undefined && summary?.flooded_pct_5ft !== null ? `${formatNumber(summary.flooded_pct_5ft)}%` : '--'}</dd>
              </div>
              <div>
                <dt>7ft exposure</dt>
                <dd>{summary?.flooded_pct_7ft !== undefined && summary?.flooded_pct_7ft !== null ? `${formatNumber(summary.flooded_pct_7ft)}%` : '--'}</dd>
              </div>
              <div>
                <dt>Series starts</dt>
                <dd>{summary?.date_min || '--'}</dd>
              </div>
              <div>
                <dt>Series ends</dt>
                <dd>{summary?.date_max || '--'}</dd>
              </div>
            </dl>
          </article>

          <article className="card detail-card detail-card-wide">
            <h2>Climate and resilience context</h2>
            <dl className="detail-list">
              <div>
                <dt>Municipalities in scope</dt>
                <dd>{formatInteger(summary?.municipality_count)}</dd>
              </div>
              <div>
                <dt>Resilient municipalities</dt>
                <dd>{formatInteger(summary?.resilient_muni_count)}</dd>
              </div>
              <div>
                <dt>Resilient share</dt>
                <dd>{formatPercent(summary?.resilient_share_pct)}</dd>
              </div>
              <div>
                <dt>Average CRS class</dt>
                <dd>{formatNumber(summary?.avg_crs_class)}</dd>
              </div>
              <div>
                <dt>Average CRS discount</dt>
                <dd>{formatPercent(summary?.avg_crs_discount_pct)}</dd>
              </div>
              <div>
                <dt>Average 2ft flood exposure</dt>
                <dd>{formatPercent(summary?.avg_flooded_pct_2ft)}</dd>
              </div>
              <div>
                <dt>Average 5ft flood exposure</dt>
                <dd>{formatPercent(summary?.avg_flooded_pct_5ft)}</dd>
              </div>
              <div>
                <dt>Average 7ft flood exposure</dt>
                <dd>{formatPercent(summary?.avg_flooded_pct_7ft)}</dd>
              </div>
              <div>
                <dt>Median municipality income</dt>
                <dd>{summary?.median_income_median ? `$${formatInteger(summary.median_income_median)}` : '--'}</dd>
              </div>
            </dl>
          </article>
        </section>
      </main>
    </div>
  )
}

export default App
