import { useEffect, useId, useMemo, useState } from 'react'
import './App.css'

const API_BASE = 'http://127.0.0.1:8000'

const MAP_METRICS = [
  ['avg_premium_bps', 'Average premium'],
  ['crs_class', 'CRS class'],
  ['flooded_pct_2ft', '2ft flood'],
  ['flooded_pct_5ft', '5ft flood'],
  ['flooded_pct_7ft', '7ft flood'],
]

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

function hexToRgb(hex) {
  const cleaned = hex.replace('#', '')
  return {
    r: parseInt(cleaned.slice(0, 2), 16),
    g: parseInt(cleaned.slice(2, 4), 16),
    b: parseInt(cleaned.slice(4, 6), 16),
  }
}

function interpolateColor(startHex, endHex, ratio) {
  const clamped = Math.max(0, Math.min(1, ratio))
  const start = hexToRgb(startHex)
  const end = hexToRgb(endHex)
  const r = Math.round(start.r + (end.r - start.r) * clamped)
  const g = Math.round(start.g + (end.g - start.g) * clamped)
  const b = Math.round(start.b + (end.b - start.b) * clamped)
  return `rgb(${r}, ${g}, ${b})`
}

function formatMapValue(metric, value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '--'
  }
  if (metric === 'avg_premium_bps') {
    return `${formatNumber(value)} bps`
  }
  if (metric === 'crs_class') {
    return `Class ${formatInteger(value)}`
  }
  return formatPercent(value)
}

function buildPathFromGeometry(geometry, projectPoint) {
  if (!geometry) {
    return ''
  }

  const ringsForPolygon = (polygon) =>
    polygon
      .map((ring) =>
        ring
          .map((point, index) => {
            const [x, y] = projectPoint(point)
            return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`
          })
          .join(' ')
      )
      .map((segment) => `${segment} Z`)
      .join(' ')

  if (geometry.type === 'Polygon') {
    return ringsForPolygon(geometry.coordinates)
  }

  if (geometry.type === 'MultiPolygon') {
    return geometry.coordinates.map((polygon) => ringsForPolygon(polygon)).join(' ')
  }

  return ''
}

function MunicipalityMap({ featureCollection, metric, selectedMunicipality, onSelectMunicipality }) {
  const width = 920
  const height = 430
  const padding = 18

  const prepared = useMemo(() => {
    if (!featureCollection?.features?.length) {
      return null
    }

    let minX = Number.POSITIVE_INFINITY
    let maxX = Number.NEGATIVE_INFINITY
    let minY = Number.POSITIVE_INFINITY
    let maxY = Number.NEGATIVE_INFINITY

    const collectBounds = (coordinates) => {
      if (!Array.isArray(coordinates)) {
        return
      }
      if (coordinates.length && typeof coordinates[0] === 'number') {
        const [x, y] = coordinates
        if (x < minX) minX = x
        if (x > maxX) maxX = x
        if (y < minY) minY = y
        if (y > maxY) maxY = y
        return
      }
      coordinates.forEach(collectBounds)
    }

    featureCollection.features.forEach((feature) => collectBounds(feature.geometry?.coordinates))
    if (!Number.isFinite(minX) || !Number.isFinite(maxX) || !Number.isFinite(minY) || !Number.isFinite(maxY)) {
      return null
    }

    const spanX = Math.max(maxX - minX, 0.0001)
    const spanY = Math.max(maxY - minY, 0.0001)
    const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY)
    const xOffset = (width - spanX * scale) / 2
    const yOffset = (height - spanY * scale) / 2

    const projectPoint = ([x, y]) => {
      const svgX = xOffset + (x - minX) * scale
      const svgY = height - (yOffset + (y - minY) * scale)
      return [svgX, svgY]
    }

    const values = featureCollection.features
      .map((feature) => feature.properties?.[metric])
      .filter((value) => value !== null && value !== undefined && !Number.isNaN(Number(value)))
      .map(Number)

    const minValue = values.length ? Math.min(...values) : 0
    const maxValue = values.length ? Math.max(...values) : 1

    const getFill = (value) => {
      if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return '#ffffff'
      }
      const numericValue = Number(value)
      const normalized =
        metric === 'crs_class'
          ? (maxValue - numericValue) / Math.max(maxValue - minValue, 1)
          : (numericValue - minValue) / Math.max(maxValue - minValue, 1)
      return interpolateColor('#D1E6F1', '#006F85', normalized)
    }

    return {
      minValue,
      maxValue,
      features: featureCollection.features.map((feature) => ({
        ...feature,
        path: buildPathFromGeometry(feature.geometry, projectPoint),
        fill: getFill(feature.properties?.[metric]),
      })),
    }
  }, [featureCollection, metric])

  if (!prepared) {
    return (
      <div className="chart-empty">
        <p>Map geometry was not available.</p>
      </div>
    )
  }

  const metricLabel = MAP_METRICS.find(([key]) => key === metric)?.[1] || metric

  return (
    <div className="map-shell">
      <div className="map-layout">
        <svg viewBox={`0 0 ${width} ${height}`} className="map-svg" role="img" aria-label={`New Jersey municipality map colored by ${metricLabel}`}>
          {prepared.features.map((feature) => {
            const isSelected = feature.properties?.mun === selectedMunicipality
            return (
              <path
                key={feature.properties?.mun}
                d={feature.path}
                fill={feature.fill}
                className={isSelected ? 'muni-path selected' : 'muni-path'}
                onClick={() => onSelectMunicipality(feature.properties?.mun)}
              >
                <title>
                  {`${feature.properties?.mun_label || feature.properties?.mun}: ${formatMapValue(metric, feature.properties?.[metric])}`}
                </title>
              </path>
            )
          })}
        </svg>

        <aside className="map-side-panel">
          <div className="map-legend">
            <span className="map-legend-title">{metricLabel}</span>
            <div className="map-legend-bar" />
            <div className="map-legend-scale">
              <span>{formatMapValue(metric, prepared.minValue)}</span>
              <span>{formatMapValue(metric, prepared.maxValue)}</span>
            </div>
          </div>

          <div className="map-side-copy">
            <strong>Map tip</strong>
            <p>Click a municipality to open its graph view and details.</p>
          </div>
        </aside>
      </div>
    </div>
  )
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
  const [displayMode, setDisplayMode] = useState('graph')
  const [level, setLevel] = useState('municipality')
  const [view, setView] = useState('weekly')
  const [metricView, setMetricView] = useState('premium')
  const [mapMetric, setMapMetric] = useState('avg_premium_bps')
  const [selectedMunicipality, setSelectedMunicipality] = useState('')
  const [selectedCounty, setSelectedCounty] = useState('')
  const [summary, setSummary] = useState(null)
  const [series, setSeries] = useState([])
  const [mapData, setMapData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([fetchJson(`${API_BASE}/api/options`), fetchJson(`${API_BASE}/api/map`)])
      .then(([optionsData, mapPayload]) => {
        setOptions(optionsData)
        setMapData(mapPayload)
        setSelectedMunicipality(optionsData.default_municipality || optionsData.municipalities[0] || '')
        setSelectedCounty(optionsData.default_county || optionsData.counties[0] || '')
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

  const jumpToMunicipalityGraph = (mun) => {
    if (!mun) {
      return
    }
    setSelectedMunicipality(mun)
    setLevel('municipality')
    setDisplayMode('graph')
  }

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
          <div className="toggle-row toggle-row-start">
            <div className="segmented-control">
              {[
                ['graph', 'Graph'],
                ['map', 'Map'],
              ].map(([option, label]) => (
                <button
                  key={option}
                  type="button"
                  className={displayMode === option ? 'segment active' : 'segment'}
                  onClick={() => setDisplayMode(option)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {displayMode === 'graph' ? (
            <>
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
                  <strong>{level === 'state' ? 'New Jersey' : activeName || 'Waiting for selection'}</strong>
                </div>
              </div>
            </>
          ) : (
            <div className="map-toolbar">
              <div className="segmented-control segmented-control-wrap">
                {MAP_METRICS.map(([option, label]) => (
                  <button
                    key={option}
                    type="button"
                    className={mapMetric === option ? 'segment active' : 'segment'}
                    onClick={() => setMapMetric(option)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>

        {error ? (
          <section className="card error-card">
            <h2>Backend connection error</h2>
            <p>{error}</p>
          </section>
        ) : null}

        {displayMode === 'graph' ? (
          <>
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
          </>
        ) : (
          <section className="card chart-card">
            <div className="section-header">
              <div>
                <h2>Municipality map view</h2>
                <p>Toggle CRS, flood exposure, and average premium layers across all New Jersey municipalities.</p>
              </div>
            </div>
            <MunicipalityMap
              featureCollection={mapData}
              metric={mapMetric}
              selectedMunicipality={selectedMunicipality}
              onSelectMunicipality={jumpToMunicipalityGraph}
            />
          </section>
        )}
      </main>
    </div>
  )
}

export default App
