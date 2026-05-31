import { useEffect, useMemo, useRef, useState } from 'react'
import { formatCurrency } from '../../lib/utils'
import type { CardAnalysis, CardTransaction } from '../../types'

type CardAnalysisPanelProps = {
  analysis: CardAnalysis | null
  error: string | null
  isLoading: boolean
  onSelectTransaction: (transactionId: string) => void
  reviewableTransactionIds: Set<string>
  transactionId: string
}

type CountryPoint = {
  country: string
  count: number
  lat: number
  lon: number
}

declare global {
  interface Window {
    __googleMapsPromise?: Promise<void>
    google?: any
  }
}

const GOOGLE_MAPS_API_KEY =
  (((import.meta as any).env?.VITE_GOOGLE_MAPS_API_KEY as string | undefined) ??
    ((import.meta as any).env?.GOOGLE_MAPS_API_KEY as string | undefined) ??
    '')

const COUNTRY_COORDS: Record<string, { lat: number; lon: number }> = {
  US: { lat: 38, lon: -97 },
  CA: { lat: 56, lon: -106 },
  MX: { lat: 23, lon: -102 },
  BR: { lat: -14, lon: -51 },
  AR: { lat: -34, lon: -64 },
  GB: { lat: 55, lon: -3 },
  FR: { lat: 46, lon: 2 },
  DE: { lat: 51, lon: 10 },
  ES: { lat: 40, lon: -4 },
  IT: { lat: 42, lon: 12 },
  NL: { lat: 52, lon: 5 },
  SE: { lat: 60, lon: 18 },
  NO: { lat: 60, lon: 8 },
  CH: { lat: 47, lon: 8 },
  PL: { lat: 52, lon: 19 },
  RU: { lat: 61, lon: 105 },
  TR: { lat: 39, lon: 35 },
  IN: { lat: 21, lon: 78 },
  CN: { lat: 35, lon: 104 },
  JP: { lat: 36, lon: 138 },
  KR: { lat: 36, lon: 128 },
  SG: { lat: 1.35, lon: 103.8 },
  AU: { lat: -25, lon: 133 },
  NZ: { lat: -41, lon: 174 },
  AE: { lat: 24, lon: 54 },
  SA: { lat: 23, lon: 45 },
  ZA: { lat: -30, lon: 24 },
  NG: { lat: 9, lon: 8 },
  EG: { lat: 26, lon: 30 },
}

export function CardAnalysisPanel({
  analysis,
  error,
  isLoading,
  onSelectTransaction,
  reviewableTransactionIds,
  transactionId,
}: CardAnalysisPanelProps) {
  if (error) {
    return (
      <section className="card-analysis-panel">
        <p className="analysis-state">{error}</p>
      </section>
    )
  }

  if (isLoading || !analysis) {
    return (
      <section className="card-analysis-panel">
        <p className="analysis-state">Loading card history...</p>
      </section>
    )
  }

  return (
    <section className="card-analysis-panel">
      <div className="card-analysis-header">
        <div>
          <h3>{analysis.cardId} history</h3>
          <p>
            {analysis.summary.transactionCount} transactions ·{' '}
            {analysis.summary.fraudCount} flagged
          </p>
        </div>
        <strong>{formatCurrency(analysis.summary.totalSpend)}</strong>
      </div>

      <dl className="detail-grid compact">
        <div>
          <dt>Median</dt>
          <dd>{formatCurrency(analysis.summary.medianAmount)}</dd>
        </div>
        <div>
          <dt>Highest</dt>
          <dd>{formatCurrency(analysis.summary.highestAmount)}</dd>
        </div>
        <div>
          <dt>Countries</dt>
          <dd>{analysis.summary.countries.join(', ') || 'None'}</dd>
        </div>
        <div>
          <dt>Categories</dt>
          <dd>{analysis.summary.categories.join(', ') || 'None'}</dd>
        </div>
      </dl>

      <CardGeoMap transactions={analysis.transactions} />

      <CardHistoryChart
        currentTransactionId={transactionId}
        onSelectTransaction={onSelectTransaction}
        reviewableTransactionIds={reviewableTransactionIds}
        transactions={analysis.transactions}
      />
    </section>
  )
}

function CardGeoMap({ transactions }: { transactions: CardTransaction[] }) {
  const points = useMemo(() => {
    const counts = useCountryCounts(transactions)

    return counts
      .map(({ country, count }) => {
        const coord = COUNTRY_COORDS[country]

        if (!coord) {
          return null
        }

        return {
          country,
          count,
          lat: coord.lat,
          lon: coord.lon,
        } as CountryPoint
      })
      .filter((value): value is CountryPoint => value !== null)
      .sort((a, b) => b.count - a.count)
  }, [transactions])

  const mapNodeRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<any>(null)
  const overlaysRef = useRef<any[]>([])
  const [mapError, setMapError] = useState<string | null>(null)

  useEffect(() => {
    if (!GOOGLE_MAPS_API_KEY) {
      setMapError('Set VITE_GOOGLE_MAPS_API_KEY in frontend/.env to load map.')
      return
    }

    if (points.length === 0) {
      setMapError(null)
      return
    }

    let cancelled = false

    loadGoogleMapsScript(GOOGLE_MAPS_API_KEY)
      .then(() => {
        if (cancelled || !mapNodeRef.current || !window.google?.maps) {
          return
        }

        setMapError(null)

        if (!mapRef.current) {
          mapRef.current = new window.google.maps.Map(mapNodeRef.current, {
            center: { lat: 20, lng: 0 },
            disableDefaultUI: true,
            gestureHandling: 'cooperative',
            mapTypeControl: false,
            streetViewControl: false,
            zoom: 2,
            zoomControl: true,
          })
        }

        for (const overlay of overlaysRef.current) {
          overlay.setMap(null)
        }
        overlaysRef.current = []

        const bounds = new window.google.maps.LatLngBounds()
        const maxCount = Math.max(1, ...points.map((point) => point.count))

        for (const point of points) {
          const center = { lat: point.lat, lng: point.lon }
          const intensity = point.count / maxCount

          const circle = new window.google.maps.Circle({
            center,
            fillColor: '#ff0080',
            fillOpacity: 0.18 + intensity * 0.55,
            map: mapRef.current,
            radius: 60000 + intensity * 340000,
            strokeColor: '#ff0080',
            strokeOpacity: 0.7,
            strokeWeight: 1,
          })

          const marker = new window.google.maps.Marker({
            map: mapRef.current,
            position: center,
            title: `${point.country}: ${point.count} transactions`,
          })

          bounds.extend(center)
          overlaysRef.current.push(circle, marker)
        }

        if (!bounds.isEmpty()) {
          mapRef.current.fitBounds(bounds, 40)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setMapError('Could not load Google Maps API for this view.')
        }
      })

    return () => {
      cancelled = true
    }
  }, [points])

  const topCountries = points.slice(0, 6)

  return (
    <section className="geo-usage-panel" aria-label="Card country usage map">
      <div className="chart-title-row">
        <strong>Card usage by country</strong>
        <span>Google Maps heat overlay by merchant country</span>
      </div>

      <div className="geo-usage-layout">
        <div className="geo-map-frame">
          {mapError ? (
            <p className="analysis-state geo-map-state">{mapError}</p>
          ) : points.length === 0 ? (
            <p className="analysis-state geo-map-state">
              No country data for this card.
            </p>
          ) : (
            <div className="geo-map-canvas" ref={mapNodeRef} />
          )}
        </div>

        <div className="geo-country-list">
          {topCountries.length === 0 ? (
            <p className="empty-copy">No country data for this card.</p>
          ) : (
            topCountries.map(({ country, count }) => {
              const width = Math.max(
                8,
                Math.round((count / Math.max(1, topCountries[0].count)) * 100),
              )

              return (
                <div className="geo-country-row" key={country}>
                  <div className="geo-country-labels">
                    <span>{country}</span>
                    <strong>{count}</strong>
                  </div>
                  <div className="geo-country-track" aria-hidden="true">
                    <span style={{ width: `${width}%` }} />
                  </div>
                </div>
              )
            })
          )}
        </div>
      </div>
    </section>
  )
}

function loadGoogleMapsScript(apiKey: string): Promise<void> {
  if (typeof window === 'undefined') {
    return Promise.resolve()
  }

  if (window.google?.maps) {
    return Promise.resolve()
  }

  if (window.__googleMapsPromise) {
    return window.__googleMapsPromise
  }

  window.__googleMapsPromise = new Promise((resolve, reject) => {
    const script = document.createElement('script')
    script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}`
    script.async = true
    script.defer = true
    script.onload = () => resolve()
    script.onerror = () => reject(new Error('Failed to load Google Maps API'))
    document.head.appendChild(script)
  })

  return window.__googleMapsPromise
}

function useCountryCounts(transactions: CardTransaction[]) {
  const map = new Map<string, number>()

  for (const transaction of transactions) {
    const country = transaction.merchantCountry?.trim().toUpperCase()
    if (!country) {
      continue
    }

    map.set(country, (map.get(country) ?? 0) + 1)
  }

  return Array.from(map.entries())
    .map(([country, count]) => ({ country, count }))
    .sort((a, b) => b.count - a.count)
}

function CardHistoryChart({
  currentTransactionId,
  onSelectTransaction,
  reviewableTransactionIds,
  transactions,
}: {
  currentTransactionId: string
  onSelectTransaction: (transactionId: string) => void
  reviewableTransactionIds: Set<string>
  transactions: CardTransaction[]
}) {
  const maxAmount = Math.max(1, ...transactions.map((item) => item.amount))
  const halfAmount = maxAmount / 2
  const firstTransaction = transactions[0]
  const lastTransaction = transactions[transactions.length - 1]

  return (
    <div className="card-history-chart" aria-label="Card transaction history">
      <div className="chart-title-row">
        <strong>Transaction amount over time</strong>
        <span>Y: amount · X: transaction date</span>
      </div>

      <div className="chart-plot">
        <div className="chart-y-axis" aria-hidden="true">
          <span>{formatCompactCurrency(maxAmount)}</span>
          <span>{formatCompactCurrency(halfAmount)}</span>
          <span>$0</span>
        </div>

        <div className="chart-body">
          <div className="chart-grid" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div className="chart-bars">
            {transactions.map((item) => {
              const amountPercent = Math.max(4, (item.amount / maxAmount) * 100)
              const canOpenTransaction = reviewableTransactionIds.has(
                item.transactionId,
              )

              return (
                <div className="chart-column" key={item.transactionId}>
                  <div className="chart-track">
                    <button
                      aria-label={`Open ${item.transactionId}`}
                      className={[
                        'chart-bar',
                        item.isFraud ? 'chart-bar-flagged' : '',
                        item.transactionId === currentTransactionId
                          ? 'chart-bar-current'
                          : '',
                        canOpenTransaction ? 'chart-bar-clickable' : '',
                      ]
                        .filter(Boolean)
                        .join(' ')}
                      disabled={!canOpenTransaction}
                      onClick={() => onSelectTransaction(item.transactionId)}
                      style={{ height: `${amountPercent}%` }}
                      title={`${item.transactionId}: ${formatCurrency(
                        item.amount,
                      )} on ${formatShortDate(item.timestamp)}${
                        canOpenTransaction ? '' : ' · not in review queue'
                      }`}
                      type="button"
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      <div className="chart-x-axis" aria-hidden="true">
        <span>
          {firstTransaction ? formatShortDate(firstTransaction.timestamp) : ''}
        </span>
        <strong>Transaction date</strong>
        <span>
          {lastTransaction ? formatShortDate(lastTransaction.timestamp) : ''}
        </span>
      </div>
      <div className="chart-legend">
        <span>
          <i className="legend-swatch legend-swatch-normal" />
          Normal transaction
        </span>
        <span>
          <i className="legend-swatch legend-swatch-flagged" />
          Flagged transaction
        </span>
        <span>
          <i className="legend-swatch legend-swatch-current" />
          Current transaction
        </span>
        <span>
          <i className="legend-swatch legend-swatch-clickable" />
          Clickable in review queue
        </span>
      </div>
    </div>
  )
}

function formatShortDate(timestamp: string) {
  const date = new Date(timestamp)

  if (Number.isNaN(date.getTime())) {
    return timestamp
  }

  return date.toLocaleDateString(undefined, {
    day: 'numeric',
    month: 'short',
  })
}

function formatCompactCurrency(value: number) {
  if (value >= 1000) {
    return `$${Math.round(value / 100) / 10}k`
  }

  return `$${Math.round(value)}`
}
