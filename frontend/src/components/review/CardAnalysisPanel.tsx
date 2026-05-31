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

      <CardGeoHeatmap transactions={analysis.transactions} />

      <CardHistoryChart
        currentTransactionId={transactionId}
        onSelectTransaction={onSelectTransaction}
        reviewableTransactionIds={reviewableTransactionIds}
        transactions={analysis.transactions}
      />
    </section>
  )
}

function CardGeoHeatmap({ transactions }: { transactions: CardTransaction[] }) {
  const counts = useCountryCounts(transactions)
  const points = counts
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

  const maxCount = Math.max(1, ...points.map((point) => point.count))
  const topCountries = counts.slice(0, 6)

  return (
    <section className="geo-usage-panel" aria-label="Card country usage heatmap">
      <div className="chart-title-row">
        <strong>Card usage by country</strong>
        <span>Heat map by merchant country frequency</span>
      </div>

      <div className="geo-usage-layout">
        <div className="geo-map-frame">
          <svg className="geo-map" viewBox="0 0 1000 500" role="img">
            <title>World map heat points for this card</title>
            <rect x="0" y="0" width="1000" height="500" className="geo-sea" />
            <g className="geo-grid" aria-hidden="true">
              <line x1="0" y1="125" x2="1000" y2="125" />
              <line x1="0" y1="250" x2="1000" y2="250" />
              <line x1="0" y1="375" x2="1000" y2="375" />
              <line x1="250" y1="0" x2="250" y2="500" />
              <line x1="500" y1="0" x2="500" y2="500" />
              <line x1="750" y1="0" x2="750" y2="500" />
            </g>
            {points.map((point) => {
              const x = ((point.lon + 180) / 360) * 1000
              const y = ((90 - point.lat) / 180) * 500
              const intensity = point.count / maxCount
              const radius = 6 + intensity * 20

              return (
                <g key={point.country}>
                  <circle
                    cx={x}
                    cy={y}
                    r={radius}
                    className="geo-heat-dot"
                    style={{ opacity: 0.25 + intensity * 0.75 }}
                  />
                  <circle cx={x} cy={y} r="3" className="geo-heat-center" />
                </g>
              )
            })}
          </svg>
        </div>

        <div className="geo-country-list">
          {topCountries.length === 0 ? (
            <p className="empty-copy">No country data for this card.</p>
          ) : (
            topCountries.map(({ country, count }) => {
              const width = Math.max(8, Math.round((count / Math.max(1, topCountries[0].count)) * 100))
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
