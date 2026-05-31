import { CardAnalysisPanel } from './CardAnalysisPanel'
import { CrossCardNetworkPanel } from './CrossCardNetworkPanel'
import { ScoringBreakdown } from './ScoringBreakdown'
import { Button } from '../ui/button'
import { Card, CardContent, CardHeader } from '../ui/card'
import { getFeedbackReasonCode } from '../../lib/reviewFeedback'
import { formatCurrency, formatDateTime } from '../../lib/utils'
import type {
  CardAnalysis,
  DecisionFeedback,
  ReviewDecision,
  SearchFieldKey,
  TransactionFlag,
} from '../../types'

type TransactionDetailProps = {
  cardAnalysis: CardAnalysis | null
  cardAnalysisError: string | null
  isCardAnalysisLoading: boolean
  isReasonsLoading?: boolean
  reasonsLoadError?: string | null
  decisionFeedback: DecisionFeedback
  onDecide: (
    transactionId: string,
    decision: Exclude<ReviewDecision, 'pending'>,
    feedback: DecisionFeedback,
  ) => void
  onDecisionFeedbackChange: (
    transactionId: string,
    feedback: DecisionFeedback,
  ) => void
  onFilterCardCountry: (payload: { cardId: string; country: string }) => void
  onFilterByField: (payload: { field: SearchFieldKey; value: string }) => void
  onFocusRelatedTransactions: (payload: {
    label: string
    transactionIds: string[]
  }) => void
  onSelectTransaction: (transactionId: string) => void
  modelThreshold: number | null
  reviewableTransactionIds: Set<string>
  transactions: TransactionFlag[]
  transaction: TransactionFlag
  useModel: boolean
}

export function TransactionDetail({
  cardAnalysis,
  cardAnalysisError,
  isCardAnalysisLoading,
  isReasonsLoading = false,
  reasonsLoadError = null,
  decisionFeedback,
  onDecide,
  onDecisionFeedbackChange,
  onFilterCardCountry,
  onFilterByField,
  onFocusRelatedTransactions,
  onSelectTransaction,
  modelThreshold,
  reviewableTransactionIds,
  transactions,
  transaction,
  useModel,
}: TransactionDetailProps) {
  const toggleReasonCode = (reasonCode: string) => {
    const selected = decisionFeedback.reasonCodes.includes(reasonCode)
    onDecisionFeedbackChange(transaction.transactionId, {
      ...decisionFeedback,
      reasonCodes: selected
        ? decisionFeedback.reasonCodes.filter((code) => code !== reasonCode)
        : [...decisionFeedback.reasonCodes, reasonCode],
    })
  }

  const updateReasoning = (reasoning: string) => {
    onDecisionFeedbackChange(transaction.transactionId, {
      ...decisionFeedback,
      reasoning,
    })
  }

  const submitDecision = (decision: Exclude<ReviewDecision, 'pending'>) => {
    onDecide(transaction.transactionId, decision, decisionFeedback)
  }

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

        <ScoringBreakdown
          modelThreshold={modelThreshold}
          transaction={transaction}
          useModel={useModel}
        />

        <section
          aria-busy={isReasonsLoading}
          aria-label="Risk signals"
          className="reasons-section"
        >
          {isReasonsLoading ? (
            <ReasonListSkeleton />
          ) : reasonsLoadError ? (
            <p className="analysis-state">{reasonsLoadError}</p>
          ) : (
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
                  <span title="Share of what drove the score for this alert">
                    {reason.weight}%
                  </span>
                </div>
              ))}
            </div>
          )}
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

        <section className="decision-feedback" aria-label="Learning update">
          <div className="decision-feedback-header">
            <strong>Learning update</strong>
            <span>Dismiss -15% · Approve +8% · Escalate +18%</span>
          </div>
          <div className="decision-signal-list">
            {transaction.reasons.map((reason, index) => {
              const reasonCode = getFeedbackReasonCode(reason, index)
              const selected = decisionFeedback.reasonCodes.includes(reasonCode)

              return (
                <label
                  className={
                    selected
                      ? 'decision-signal-row decision-signal-row-selected'
                      : 'decision-signal-row'
                  }
                  key={reasonCode}
                >
                  <input
                    checked={selected}
                    onChange={() => toggleReasonCode(reasonCode)}
                    type="checkbox"
                  />
                  <span>
                    <strong>{reason.label}</strong>
                    <span>{reason.detail}</span>
                  </span>
                </label>
              )
            })}
          </div>
          <label className="decision-feedback-notes">
            <span>Reasoning</span>
            <textarea
              onChange={(event) => updateReasoning(event.target.value)}
              placeholder="Why this decision should change future heuristic scoring"
              rows={3}
              value={decisionFeedback.reasoning}
            />
          </label>
        </section>

        <div className="action-row">
          <Button onClick={() => submitDecision('approved')}>
            Approve
          </Button>
          <Button
            onClick={() => submitDecision('dismissed')}
            variant="outline"
          >
            Dismiss
          </Button>
          <Button
            onClick={() => submitDecision('escalated')}
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

function ReasonListSkeleton() {
  return (
    <div aria-hidden="true" className="reason-list reason-list-skeleton">
      {Array.from({ length: 3 }, (_, index) => (
        <div className="reason-skeleton-row" key={index}>
          <div className="reason-skeleton-copy">
            <span className="reason-skeleton-line reason-skeleton-line-title" />
            <span className="reason-skeleton-line reason-skeleton-line-detail" />
          </div>
          <span className="reason-skeleton-line reason-skeleton-line-weight" />
        </div>
      ))}
    </div>
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
