import { useCallback, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { EmptyTransactionDetail, TransactionDetail } from './review/TransactionDetail'
import { QueueList } from './review/QueueList'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Slider } from './ui/slider'
import { Tabs } from './ui/tabs'
import { fetchCardAnalysis, submitReviewDecision } from '../api/review'
import type { ReviewSession } from '../lib/reviewSessions'
import type {
  DecisionAction,
  ReviewDecision,
  TransactionFlag,
} from '../types'

type QueueFilter = 'pending' | 'all' | 'approved' | 'dismissed' | 'escalated'

type ReviewQueueProps = {
  activeFileHash: string
  fileHash: string
  items: TransactionFlag[]
  onReset: () => void
  onSelectSession: (fileHash: string) => void
  sessions: ReviewSession[]
}

const filterOptions: Array<{ value: QueueFilter; label: string }> = [
  { value: 'pending', label: 'Pending' },
  { value: 'all', label: 'All' },
  { value: 'approved', label: 'Approved' },
  { value: 'dismissed', label: 'Dismissed' },
  { value: 'escalated', label: 'Escalated' },
]

const shortcutOptions = [
  { keys: 'J / ↓', label: 'Next' },
  { keys: 'K / ↑', label: 'Previous' },
  { keys: 'A', label: 'Approve' },
  { keys: 'D', label: 'Dismiss' },
  { keys: 'E', label: 'Escalate' },
  { keys: 'U', label: 'Undo' },
]

export function ReviewQueue({
  activeFileHash,
  fileHash,
  items,
  onReset,
  onSelectSession,
  sessions,
}: ReviewQueueProps) {
  const [transactions, setTransactions] = useState(items)
  const [activeId, setActiveId] = useState(items[0]?.transactionId ?? '')
  const [filter, setFilter] = useState<QueueFilter>('pending')
  const [query, setQuery] = useState('')
  const [threshold, setThreshold] = useState(55)
  const [history, setHistory] = useState<DecisionAction[]>([])
  const [selectedCardId, setSelectedCardId] = useState<string | null>(null)
  const {
    isError: reviewSyncFailed,
    mutate: syncReviewDecision,
  } = useMutation({
    mutationFn: submitReviewDecision,
  })
  const selectedCardAnalysisQuery = useQuery({
    enabled: Boolean(selectedCardId),
    queryFn: () =>
      fetchCardAnalysis({
        cardId: selectedCardId ?? '',
        fileHash,
      }),
    queryKey: ['card-analysis', fileHash, selectedCardId],
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

  useEffect(() => {
    setTransactions(items)
    setActiveId(items[0]?.transactionId ?? '')
    setFilter('pending')
    setHistory([])
    setSelectedCardId(null)
  }, [fileHash, items])

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
    <div className="app-shell review-shell">
      <main className="workspace">
        <header className="topbar">
          <div className="review-status" aria-label="Review status">
            <strong>{queueStats.pending}</strong>
            <span>pending</span>
            <span>{visibleTransactions.length} shown</span>
            <span>{transactions.length} flagged</span>
            {reviewSyncFailed ? <span>Sync failed</span> : null}
          </div>
          <div className="topbar-actions">
            {sessions.length > 1 ? (
              <select
                aria-label="Switch result set"
                className="result-select"
                onChange={(event) => onSelectSession(event.target.value)}
                value={activeFileHash}
              >
                {sessions.map((session) => (
                  <option key={session.fileHash} value={session.fileHash}>
                    {session.label} · {session.fileHash.slice(0, 8)}
                  </option>
                ))}
              </select>
            ) : null}
            <Input
              aria-label="Search transactions"
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search transaction"
              value={query}
            />
            <Button
              disabled={history.length === 0}
              onClick={undo}
              size="sm"
              variant="outline"
            >
              Undo
            </Button>
            <Button onClick={onReset} size="sm" variant="outline">
              Upload CSV
            </Button>
          </div>
        </header>

        <div className="review-controls">
          <Tabs
            className="queue-tabs"
            onValueChange={setFilter}
            options={filterOptions}
            value={filter}
          />
          <label className="threshold-control" htmlFor="review-threshold">
            <span>Threshold</span>
            <Slider
              id="review-threshold"
              max={95}
              min={20}
              onChange={(event) => setThreshold(Number(event.target.value))}
              value={threshold}
            />
            <strong>{threshold}%</strong>
          </label>
        </div>

        <div className="shortcut-strip" aria-label="Keyboard shortcuts">
          {shortcutOptions.map((shortcut) => (
            <span className="shortcut-token" key={shortcut.label}>
              <kbd>{shortcut.keys}</kbd>
              <span>{shortcut.label}</span>
            </span>
          ))}
        </div>

        <div className="review-layout">
          <QueueList
            activeTransactionId={activeTransaction?.transactionId}
            onSelect={setActiveId}
            transactions={visibleTransactions}
          />

          {activeTransaction ? (
            <TransactionDetail
              cardAnalysis={
                selectedCardId === activeTransaction.cardId
                  ? selectedCardAnalysisQuery.data ?? null
                  : null
              }
              cardAnalysisError={
                selectedCardId === activeTransaction.cardId &&
                selectedCardAnalysisQuery.error instanceof Error
                  ? selectedCardAnalysisQuery.error.message
                  : null
              }
              isCardAnalysisLoading={
                selectedCardId === activeTransaction.cardId &&
                selectedCardAnalysisQuery.isFetching
              }
              onDecide={decide}
              onSelectCard={setSelectedCardId}
              selectedCardId={selectedCardId}
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
