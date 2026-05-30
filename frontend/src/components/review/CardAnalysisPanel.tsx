import { formatCurrency } from '../../lib/utils'
import type { CardAnalysis, CardTransaction } from '../../types'

type CardAnalysisPanelProps = {
  analysis: CardAnalysis | null
  error: string | null
  isLoading: boolean
  transactionId: string
}

export function CardAnalysisPanel({
  analysis,
  error,
  isLoading,
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

      <CardHistoryChart
        currentTransactionId={transactionId}
        transactions={analysis.transactions}
      />
    </section>
  )
}

function CardHistoryChart({
  currentTransactionId,
  transactions,
}: {
  currentTransactionId: string
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

              return (
                <div className="chart-column" key={item.transactionId}>
                  <div className="chart-track">
                    <span
                      className={[
                        'chart-bar',
                        item.isFraud ? 'chart-bar-flagged' : '',
                        item.transactionId === currentTransactionId
                          ? 'chart-bar-current'
                          : '',
                      ]
                        .filter(Boolean)
                        .join(' ')}
                      style={{ height: `${amountPercent}%` }}
                      title={`${item.transactionId}: ${formatCurrency(
                        item.amount,
                      )} on ${formatShortDate(item.timestamp)}`}
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
