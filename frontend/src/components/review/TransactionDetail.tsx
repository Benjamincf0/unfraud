import { CardAnalysisPanel } from './CardAnalysisPanel'
import { CrossCardNetworkPanel } from './CrossCardNetworkPanel'
import { Button } from '../ui/button'
import { Card, CardContent, CardHeader } from '../ui/card'
import { formatCurrency, formatDateTime } from '../../lib/utils'
import type {
  CardAnalysis,
  ReviewDecision,
  SearchFieldKey,
  TransactionFlag,
} from '../../types'

type TransactionDetailProps = {
  cardAnalysis: CardAnalysis | null
  cardAnalysisError: string | null
  isCardAnalysisLoading: boolean
  onDecide: (
    transactionId: string,
    decision: Exclude<ReviewDecision, 'pending'>,
  ) => void
  onFilterCardCountry: (payload: { cardId: string; country: string }) => void
  onFilterByField: (payload: { field: SearchFieldKey; value: string }) => void
  onFocusRelatedTransactions: (payload: {
    label: string
    transactionIds: string[]
  }) => void
  onSelectTransaction: (transactionId: string) => void
  reviewableTransactionIds: Set<string>
  transactions: TransactionFlag[]
  transaction: TransactionFlag
}

export function TransactionDetail({
  cardAnalysis,
  cardAnalysisError,
  isCardAnalysisLoading,
  onDecide,
  onFilterCardCountry,
  onFilterByField,
  onFocusRelatedTransactions,
  onSelectTransaction,
  reviewableTransactionIds,
  transactions,
  transaction,
}: TransactionDetailProps) {
  return (
    <Card className="transaction-detail">
      <CardHeader>
        <div>
          <h2>
            <FieldLink
              field="merchant_name"
              onFilterByField={onFilterByField}
              value={transaction.merchantName}
            />
          </h2>
          <p>
            <FieldLink
              field="transaction_id"
              onFilterByField={onFilterByField}
              value={transaction.transactionId}
            />
            <span> · </span>
            <FieldLink
              displayValue={formatDateTime(transaction.timestamp)}
              field="timestamp"
              onFilterByField={onFilterByField}
              value={transaction.timestamp}
            />
          </p>
        </div>
        <span className="risk-label">{transaction.label}</span>
      </CardHeader>

      <CardContent>
        <div className="amount-row">
          <span>
            <FieldLink
              displayValue={formatCurrency(transaction.amount)}
              field="amount"
              onFilterByField={onFilterByField}
              value={String(transaction.amount)}
            />
          </span>
          <span>{Math.round(transaction.score * 100)} risk</span>
        </div>

        <dl className="detail-grid">
          <div>
            <dt>Card</dt>
            <dd>
              <FieldLink
                field="card_id"
                onFilterByField={onFilterByField}
                value={transaction.cardId}
              />
            </dd>
          </div>
          <div>
            <dt>Channel</dt>
            <dd>
              <FieldLink
                field="channel"
                onFilterByField={onFilterByField}
                value={transaction.channel}
              />
            </dd>
          </div>
          <div>
            <dt>Category</dt>
            <dd>
              <FieldLink
                field="merchant_category"
                onFilterByField={onFilterByField}
                value={transaction.merchantCategory}
              />
            </dd>
          </div>
          <div>
            <dt>Countries</dt>
            <dd>
              <FieldLink
                field="cardholder_country"
                onFilterByField={onFilterByField}
                value={transaction.cardholderCountry}
              />
              <span> to </span>
              <FieldLink
                field="merchant_country"
                onFilterByField={onFilterByField}
                value={transaction.merchantCountry}
              />
            </dd>
          </div>
          <div>
            <dt>Device</dt>
            <dd>
              {transaction.deviceId ? (
                <FieldLink
                  field="device_id"
                  onFilterByField={onFilterByField}
                  value={transaction.deviceId}
                />
              ) : (
                'Not present'
              )}
            </dd>
          </div>
          <div>
            <dt>IP</dt>
            <dd>
              {transaction.ipAddress ? (
                <FieldLink
                  field="ip_address"
                  onFilterByField={onFilterByField}
                  value={transaction.ipAddress}
                />
              ) : (
                'Not present'
              )}
            </dd>
          </div>
        </dl>

        <section className="reasons-section">
          <div className="reason-list">
            {transaction.reasons.map((reason) => (
              <div className="reason-row" key={reason.id}>
                <div>
                  <strong>
                    {reason.label}
                    {reason.signalType ? (
                      <span className="reason-signal-type">
                        {reason.signalType.replace('_', ' ')}
                      </span>
                    ) : null}
                  </strong>
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
              <dd>
                <FieldLinkList
                  field="merchant_country"
                  onFilterByField={onFilterByField}
                  values={transaction.cardContext.usualCountries}
                />
              </dd>
            </div>
            <div>
              <dt>Categories</dt>
              <dd>
                <FieldLinkList
                  field="merchant_category"
                  onFilterByField={onFilterByField}
                  values={transaction.cardContext.usualCategories}
                />
              </dd>
            </div>
          </dl>
        </section>

        <CardAnalysisPanel
          analysis={cardAnalysis}
          error={cardAnalysisError}
          isLoading={isCardAnalysisLoading}
          onFilterCardCountry={onFilterCardCountry}
          onSelectTransaction={onSelectTransaction}
          reviewableTransactionIds={reviewableTransactionIds}
          transactionId={transaction.transactionId}
        />
        <CrossCardNetworkPanel
          activeTransaction={transaction}
          onFocusRelatedTransactions={onFocusRelatedTransactions}
          transactions={transactions}
        />

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

function FieldLink({
  displayValue,
  field,
  onFilterByField,
  value,
}: {
  displayValue?: string
  field: SearchFieldKey
  onFilterByField: (payload: { field: SearchFieldKey; value: string }) => void
  value: string
}) {
  return (
    <button
      className="field-link"
      onClick={() => onFilterByField({ field, value })}
      type="button"
    >
      {displayValue ?? value}
    </button>
  )
}

function FieldLinkList({
  field,
  onFilterByField,
  values,
}: {
  field: SearchFieldKey
  onFilterByField: (payload: { field: SearchFieldKey; value: string }) => void
  values: string[]
}) {
  if (values.length === 0) {
    return <>None</>
  }

  return (
    <>
      {values.map((value, index) => (
        <span className="field-link-list-item" key={value}>
          {index > 0 ? <span>, </span> : null}
          <FieldLink
            field={field}
            onFilterByField={onFilterByField}
            value={value}
          />
        </span>
      ))}
    </>
  )
}
