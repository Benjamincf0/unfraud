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

  return (
    <div className="card-history-chart" aria-label="Card transaction history">
      <div className="chart-bars">
        {transactions.map((item) => {
          const amountPercent = Math.max(4, (item.amount / maxAmount) * 100)
          const riskPercent = Math.max(2, item.score * 100)

          return (
            <div className="chart-column" key={item.transactionId}>
              <div className="chart-track">
                <span
                  className={[
                    'chart-risk-marker',
                    item.isFraud ? 'chart-risk-marker-flagged' : '',
                  ]
                    .filter(Boolean)
                    .join(' ')}
                  style={{ bottom: `${riskPercent}%` }}
                  title={`${Math.round(item.score * 100)} risk`}
                />
                <span
                  className={[
                    'chart-bar',
                    item.transactionId === currentTransactionId
                      ? 'chart-bar-current'
                      : '',
                  ]
                    .filter(Boolean)
                    .join(' ')}
                  style={{ height: `${amountPercent}%` }}
                  title={`${item.transactionId}: ${formatCurrency(item.amount)}`}
                />
              </div>
            </div>
          )
        })}
      </div>
      <div className="chart-legend">
        <span>Amount</span>
        <span>Risk marker</span>
        <span>Current transaction</span>
      </div>
    </div>
  )
}
