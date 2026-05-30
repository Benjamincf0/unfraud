import { useCallback, useEffect, useMemo, useState } from 'react'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Card, CardContent, CardHeader } from './ui/card'
import { Input } from './ui/input'
import { Slider } from './ui/slider'
import { Tabs } from './ui/tabs'
import { formatCurrency, formatDateTime } from '../lib/utils'
import type {
  AuditEntry,
  DecisionAction,
  ReviewDecision,
  TransactionFlag,
} from '../types'

type QueueFilter = 'pending' | 'all' | 'approved' | 'dismissed' | 'escalated'

type ReviewQueueProps = {
  items: TransactionFlag[]
}

const filterOptions: Array<{ value: QueueFilter; label: string }> = [
  { value: 'pending', label: 'Pending' },
  { value: 'all', label: 'All' },
  { value: 'approved', label: 'Approve' },
  { value: 'dismissed', label: 'Dismiss' },
  { value: 'escalated', label: 'Escalate' },
]

export function ReviewQueue({ items }: ReviewQueueProps) {
  const [transactions, setTransactions] = useState(items)
  const [activeId, setActiveId] = useState(items[0]?.transactionId ?? '')
  const [filter, setFilter] = useState<QueueFilter>('pending')
  const [query, setQuery] = useState('')
  const [threshold, setThreshold] = useState(55)
  const [history, setHistory] = useState<DecisionAction[]>([])
  const [auditLog, setAuditLog] = useState<AuditEntry[]>([])

  const visibleTransactions = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()

    return transactions.filter((transaction) => {
      const scorePasses = transaction.score * 100 >= threshold
      const filterPasses =
        filter === 'all' ? true : transaction.decision === filter
      const queryPasses = normalizedQuery
        ? [
            transaction.transactionId,
            transaction.cardId,
            transaction.merchantName,
            transaction.merchantCategory,
            transaction.merchantCountry,
          ]
            .join(' ')
            .toLowerCase()
            .includes(normalizedQuery)
        : true

      return scorePasses && filterPasses && queryPasses
    })
  }, [filter, query, threshold, transactions])

  const activeTransaction =
    visibleTransactions.find((transaction) => transaction.transactionId === activeId) ??
    visibleTransactions[0]

  const queueStats = useMemo(() => {
    return transactions.reduce(
      (stats, transaction) => {
        stats[transaction.decision] += 1
        return stats
      },
      { approved: 0, dismissed: 0, escalated: 0, pending: 0 } as Record<
        ReviewDecision,
        number
      >,
    )
  }, [transactions])

  const decide = useCallback((
    transactionId: string,
    nextDecision: Exclude<ReviewDecision, 'pending'>,
  ) => {
    setTransactions((current) =>
      current.map((transaction) => {
        if (transaction.transactionId !== transactionId) {
          return transaction
        }

        setHistory((previous) => [
          {
            transactionId,
            previousDecision: transaction.decision,
            nextDecision,
          },
          ...previous,
        ])

        setAuditLog((previous) => [
          {
            id: `${transactionId}-${Date.now()}`,
            transactionId,
            decision: nextDecision,
            timestamp: new Date().toISOString(),
          },
          ...previous,
        ])

        return { ...transaction, decision: nextDecision }
      }),
    )
  }, [])

  const undo = useCallback(() => {
    const [lastAction, ...rest] = history

    if (!lastAction) {
      return
    }

    setTransactions((current) =>
      current.map((transaction) =>
        transaction.transactionId === lastAction.transactionId
          ? { ...transaction, decision: lastAction.previousDecision }
          : transaction,
      ),
    )
    setHistory(rest)
    setAuditLog((current) => current.slice(1))
    setActiveId(lastAction.transactionId)
  }, [history])

  const moveActive = useCallback((direction: 1 | -1) => {
    if (!activeTransaction) {
      return
    }

    const activeIndex = visibleTransactions.findIndex(
      (transaction) => transaction.transactionId === activeTransaction.transactionId,
    )
    const nextIndex = Math.min(
      Math.max(activeIndex + direction, 0),
      visibleTransactions.length - 1,
    )
    setActiveId(visibleTransactions[nextIndex]?.transactionId ?? '')
  }, [activeTransaction, visibleTransactions])

  useEffect(() => {
    if (activeTransaction) {
      setActiveId(activeTransaction.transactionId)
    }
  }, [activeTransaction])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null

      if (target?.matches('input, textarea, select')) {
        return
      }

      if (event.key === 'j' || event.key === 'ArrowDown') {
        event.preventDefault()
        moveActive(1)
      }

      if (event.key === 'k' || event.key === 'ArrowUp') {
        event.preventDefault()
        moveActive(-1)
      }

      if (!activeTransaction) {
        return
      }

      if (event.key.toLowerCase() === 'a') {
        decide(activeTransaction.transactionId, 'approved')
      }

      if (event.key.toLowerCase() === 'd') {
        decide(activeTransaction.transactionId, 'dismissed')
      }

      if (event.key.toLowerCase() === 'e') {
        decide(activeTransaction.transactionId, 'escalated')
      }

      if (event.key.toLowerCase() === 'u') {
        undo()
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [activeTransaction, decide, moveActive, undo])

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Review summary">
        <div className="brand">
          <span className="brand-mark">FH</span>
          <div>
            <p className="brand-title">Fraud Hunter</p>
            <p className="brand-subtitle">Reviewer queue</p>
          </div>
        </div>

        <nav className="summary-list" aria-label="Queue status">
          <SummaryRow label="Pending" value={queueStats.pending} />
          <SummaryRow label="Approved" value={queueStats.approved} />
          <SummaryRow label="Dismissed" value={queueStats.dismissed} />
          <SummaryRow label="Escalated" value={queueStats.escalated} />
        </nav>

        <div className="side-section">
          <label htmlFor="cost-slider">Review threshold</label>
          <Slider
            id="cost-slider"
            max={95}
            min={20}
            onChange={(event) => setThreshold(Number(event.target.value))}
            value={threshold}
          />
          <div className="threshold-row">
            <span>{threshold}%</span>
            <span>{visibleTransactions.length} shown</span>
          </div>
        </div>

        <div className="side-section audit-section">
          <div className="section-title-row">
            <h2>Audit Trail</h2>
            <Button
              aria-label="Undo last action"
              disabled={history.length === 0}
              onClick={undo}
              size="sm"
              variant="outline"
            >
              Undo
            </Button>
          </div>
          <div className="audit-list">
            {auditLog.length === 0 ? (
              <p className="empty-copy">No decisions recorded.</p>
            ) : (
              auditLog.slice(0, 6).map((entry) => (
                <div className="audit-row" key={entry.id}>
                  <span>{entry.transactionId}</span>
                  <Badge tone={entry.decision === 'escalated' ? 'high' : 'neutral'}>
                    {entry.decision}
                  </Badge>
                </div>
              ))
            )}
          </div>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <h1>Flagged Transactions</h1>
            <p>
              {queueStats.pending} pending of {transactions.length} flagged
            </p>
          </div>
          <Input
            aria-label="Search transactions"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search card, merchant, country"
            value={query}
          />
        </header>

        <Tabs
          className="queue-tabs"
          onValueChange={setFilter}
          options={filterOptions}
          value={filter}
        />

        <div className="review-layout">
          <Card className="queue-list" aria-label="Transaction queue">
            <CardHeader>
              <h2>Queue</h2>
              <span>{visibleTransactions.length}</span>
            </CardHeader>
            <CardContent>
              {visibleTransactions.length === 0 ? (
                <p className="empty-copy">No transactions match the current view.</p>
              ) : (
                visibleTransactions.map((transaction) => (
                  <button
                    className={
                      transaction.transactionId === activeTransaction?.transactionId
                        ? 'queue-item queue-item-active'
                        : 'queue-item'
                    }
                    key={transaction.transactionId}
                    onClick={() => setActiveId(transaction.transactionId)}
                    type="button"
                  >
                    <span>
                      <strong>{transaction.merchantName}</strong>
                      <span className="queue-item-id">
                        {transaction.transactionId}
                      </span>
                    </span>
                    <span className="queue-item-meta">
                      {formatCurrency(transaction.amount)}
                      <Badge tone={transaction.label}>{Math.round(transaction.score * 100)}</Badge>
                    </span>
                  </button>
                ))
              )}
            </CardContent>
          </Card>

          {activeTransaction ? (
            <TransactionDetail
              onDecide={decide}
              transaction={activeTransaction}
            />
          ) : (
            <Card className="transaction-detail">
              <CardContent>
                <p className="empty-copy">Select a transaction to review.</p>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  )
}

function SummaryRow({ label, value }: { label: string; value: number }) {
  return (
    <div className="summary-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function TransactionDetail({
  onDecide,
  transaction,
}: {
  onDecide: (
    transactionId: string,
    decision: Exclude<ReviewDecision, 'pending'>,
  ) => void
  transaction: TransactionFlag
}) {
  return (
    <Card className="transaction-detail">
      <CardHeader>
        <div>
          <h2>{transaction.merchantName}</h2>
          <p>
            {transaction.transactionId} · {formatDateTime(transaction.timestamp)}
          </p>
        </div>
        <Badge tone={transaction.label}>{transaction.label}</Badge>
      </CardHeader>

      <CardContent>
        <div className="amount-row">
          <span>{formatCurrency(transaction.amount)}</span>
          <Badge tone={transaction.decision === 'pending' ? 'neutral' : 'success'}>
            {transaction.decision}
          </Badge>
        </div>

        <dl className="detail-grid">
          <div>
            <dt>Card</dt>
            <dd>{transaction.cardId}</dd>
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
          <h3>Reasons</h3>
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
          <h3>Card Baseline</h3>
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

        <div className="action-row">
          <Button onClick={() => onDecide(transaction.transactionId, 'approved')}>
            Approve
          </Button>
          <Button
            onClick={() => onDecide(transaction.transactionId, 'dismissed')}
            variant="secondary"
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
