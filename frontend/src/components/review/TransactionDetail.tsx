import { CardAnalysisPanel } from './CardAnalysisPanel'
import { Button } from '../ui/button'
import { Card, CardContent, CardHeader } from '../ui/card'
import { formatCurrency, formatDateTime } from '../../lib/utils'
import type { CardAnalysis, ReviewDecision, TransactionFlag } from '../../types'

type TransactionDetailProps = {
  cardAnalysis: CardAnalysis | null
  cardAnalysisError: string | null
  isCardAnalysisLoading: boolean
  onDecide: (
    transactionId: string,
    decision: Exclude<ReviewDecision, 'pending'>,
  ) => void
  onSelectCard: (cardId: string) => void
  selectedCardId: string | null
  transaction: TransactionFlag
}

export function TransactionDetail({
  cardAnalysis,
  cardAnalysisError,
  isCardAnalysisLoading,
  onDecide,
  onSelectCard,
  selectedCardId,
  transaction,
}: TransactionDetailProps) {
  const isCurrentCardSelected = selectedCardId === transaction.cardId

  return (
    <Card className="transaction-detail">
      <CardHeader>
        <div>
          <h2>{transaction.merchantName}</h2>
          <p>
            {transaction.transactionId} · {formatDateTime(transaction.timestamp)}
          </p>
        </div>
        <span className="risk-label">{transaction.label}</span>
      </CardHeader>

      <CardContent>
        <div className="amount-row">
          <span>{formatCurrency(transaction.amount)}</span>
          <span>{Math.round(transaction.score * 100)} risk</span>
        </div>

        <dl className="detail-grid">
          <div>
            <dt>Card</dt>
            <dd>
              <button
                className="inline-link"
                onClick={() => onSelectCard(transaction.cardId)}
                type="button"
              >
                {transaction.cardId}
              </button>
            </dd>
          </div>
          <div>
            <dt>Channel</dt>
            <dd>{transaction.channel}</dd>
          </div>
          <div>
            <dt>Category</dt>
            <dd>{transaction.merchantCategory}</dd>
          </div>
          <div>
            <dt>Countries</dt>
            <dd>
              {transaction.cardholderCountry} to {transaction.merchantCountry}
            </dd>
          </div>
          <div>
            <dt>Device</dt>
            <dd>{transaction.deviceId ?? 'Not present'}</dd>
          </div>
          <div>
            <dt>IP</dt>
            <dd>{transaction.ipAddress ?? 'Not present'}</dd>
          </div>
        </dl>

        <section className="reasons-section">
          <div className="reason-list">
            {transaction.reasons.map((reason) => (
              <div className="reason-row" key={reason.id}>
                <div>
                  <strong>{reason.label}</strong>
                  <p>{reason.detail}</p>
                </div>
                <span>{reason.weight}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="context-section">
          <dl className="detail-grid compact">
            <div>
              <dt>Median amount</dt>
              <dd>{formatCurrency(transaction.cardContext.medianAmount)}</dd>
            </div>
            <div>
              <dt>Transactions</dt>
              <dd>{transaction.cardContext.previousTransactions}</dd>
            </div>
            <div>
              <dt>Countries</dt>
              <dd>{transaction.cardContext.usualCountries.join(', ')}</dd>
            </div>
            <div>
              <dt>Categories</dt>
              <dd>{transaction.cardContext.usualCategories.join(', ')}</dd>
            </div>
          </dl>
        </section>

        {isCurrentCardSelected ? (
          <CardAnalysisPanel
            analysis={cardAnalysis}
            error={cardAnalysisError}
            isLoading={isCardAnalysisLoading}
            transactionId={transaction.transactionId}
          />
        ) : (
          <section className="card-analysis-prompt">
            <div>
              <strong>Card history</strong>
            </div>
            <Button
              onClick={() => onSelectCard(transaction.cardId)}
              size="sm"
              variant="outline"
            >
              Open Card
            </Button>
          </section>
        )}

        <div className="action-row">
          <Button onClick={() => onDecide(transaction.transactionId, 'approved')}>
            Approve
          </Button>
          <Button
            onClick={() => onDecide(transaction.transactionId, 'dismissed')}
            variant="outline"
          >
            Dismiss
          </Button>
          <Button
            onClick={() => onDecide(transaction.transactionId, 'escalated')}
            variant="danger"
          >
            Escalate
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

export function EmptyTransactionDetail() {
  return (
    <Card className="transaction-detail">
      <CardContent>
        <p className="empty-copy">Select a transaction to review.</p>
      </CardContent>
    </Card>
  )
}
