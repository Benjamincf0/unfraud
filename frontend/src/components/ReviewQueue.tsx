import { useCallback, useEffect, useMemo, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { EmptyTransactionDetail, TransactionDetail } from './review/TransactionDetail'
import { QueueList } from './review/QueueList'
import { ReviewSidebar } from './review/ReviewSidebar'
import { Input } from './ui/input'
import { Tabs } from './ui/tabs'
import { submitReviewDecision } from '../api/review'
import type {
  AuditEntry,
  DecisionAction,
  ReviewDecision,
  TransactionFlag,
} from '../types'

type QueueFilter = 'pending' | 'all' | 'approved' | 'dismissed' | 'escalated'

type ReviewQueueProps = {
  fileHash: string
  items: TransactionFlag[]
  onReset: () => void
}

const filterOptions: Array<{ value: QueueFilter; label: string }> = [
  { value: 'pending', label: 'Pending' },
  { value: 'all', label: 'All' },
  { value: 'approved', label: 'Approve' },
  { value: 'dismissed', label: 'Dismiss' },
  { value: 'escalated', label: 'Escalate' },
]

export function ReviewQueue({ fileHash, items, onReset }: ReviewQueueProps) {
  const [transactions, setTransactions] = useState(items)
  const [activeId, setActiveId] = useState(items[0]?.transactionId ?? '')
  const [filter, setFilter] = useState<QueueFilter>('pending')
  const [query, setQuery] = useState('')
  const [threshold, setThreshold] = useState(55)
  const [history, setHistory] = useState<DecisionAction[]>([])
  const [auditLog, setAuditLog] = useState<AuditEntry[]>([])
  const {
    isError: reviewSyncFailed,
    mutate: syncReviewDecision,
  } = useMutation({
    mutationFn: submitReviewDecision,
  })

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

  const decide = useCallback(
    (
      transactionId: string,
      nextDecision: Exclude<ReviewDecision, 'pending'>,
    ) => {
      const transaction = transactions.find(
        (item) => item.transactionId === transactionId,
      )

      if (!transaction) {
        return
      }

      const action = {
        nextDecision,
        previousDecision: transaction.decision,
        transactionId,
      }

      setTransactions((current) =>
        current.map((item) =>
          item.transactionId === transactionId
            ? { ...item, decision: nextDecision }
            : item,
        ),
      )
      setHistory((previous) => [action, ...previous])
      setAuditLog((previous) => [
        {
          id: `${transactionId}-${Date.now()}`,
          transactionId,
          decision: nextDecision,
          timestamp: new Date().toISOString(),
        },
        ...previous,
      ])
      syncReviewDecision({
        decision: nextDecision,
        fileHash,
        transactionId,
      })
    },
    [fileHash, syncReviewDecision, transactions],
  )

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

  const moveActive = useCallback(
    (direction: 1 | -1) => {
      if (!activeTransaction) {
        return
      }

      const activeIndex = visibleTransactions.findIndex(
        (transaction) =>
          transaction.transactionId === activeTransaction.transactionId,
      )
      const nextIndex = Math.min(
        Math.max(activeIndex + direction, 0),
        visibleTransactions.length - 1,
      )
      setActiveId(visibleTransactions[nextIndex]?.transactionId ?? '')
    },
    [activeTransaction, visibleTransactions],
  )

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
      <ReviewSidebar
        auditLog={auditLog}
        historyCount={history.length}
        onThresholdChange={setThreshold}
        onUndo={undo}
        shownCount={visibleTransactions.length}
        stats={queueStats}
        threshold={threshold}
      />

      <main className="workspace">
        <header className="topbar">
          <div>
            <h1>Flagged Transactions</h1>
            <p>
              {queueStats.pending} pending of {transactions.length} flagged ·{' '}
              Uploaded CSV {reviewSyncFailed ? '· Last review sync failed' : ''}
            </p>
          </div>
          <div className="topbar-actions">
            <Input
              aria-label="Search transactions"
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search card, merchant, country"
              value={query}
            />
            <button className="text-button" onClick={onReset} type="button">
              Upload another CSV
            </button>
          </div>
        </header>

        <Tabs
          className="queue-tabs"
          onValueChange={setFilter}
          options={filterOptions}
          value={filter}
        />

        <div className="review-layout">
          <QueueList
            activeTransactionId={activeTransaction?.transactionId}
            onSelect={setActiveId}
            transactions={visibleTransactions}
          />

          {activeTransaction ? (
            <TransactionDetail
              onDecide={decide}
              transaction={activeTransaction}
            />
          ) : (
            <EmptyTransactionDetail />
          )}
        </div>
      </main>
    </div>
  )
}
