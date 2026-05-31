import { useCallback, useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { EmptyTransactionDetail, TransactionDetail } from './review/TransactionDetail'
import { QueueList } from './review/QueueList'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Tabs } from './ui/tabs'
import { fetchCardAnalysis, submitReviewDecision } from '../api/review'
import type { ReviewSession } from '../lib/reviewSessions'
import type { DecisionAction, ReviewDecision, TransactionFlag } from '../types'

type QueueFilter = 'pending' | 'all' | 'approved' | 'dismissed' | 'escalated'

type SearchFieldKey =
  | 'transaction_id'
  | 'timestamp'
  | 'card_id'
  | 'amount'
  | 'merchant_name'
  | 'merchant_category'
  | 'channel'
  | 'cardholder_country'
  | 'merchant_country'
  | 'device_id'
  | 'ip_address'

type SearchMode = 'all' | 'single' | 'custom'

type ReviewQueueProps = {
  activeFileHash: string
  fileHash: string
  items: TransactionFlag[]
  onReset: () => void
  onSelectSession: (fileHash: string) => void
  sessions: ReviewSession[]
  useModel: boolean
}

const filterOptions: Array<{ value: QueueFilter; label: string }> = [
  { value: 'pending', label: 'Pending' },
  { value: 'all', label: 'All' },
  { value: 'approved', label: 'Approved' },
  { value: 'dismissed', label: 'Dismissed' },
  { value: 'escalated', label: 'Escalated' },
]

const shortcutOptions = [
  { keys: 'J / Down', label: 'Next transaction' },
  { keys: 'K / Up', label: 'Previous transaction' },
  { keys: 'A', label: 'Approve' },
  { keys: 'D', label: 'Dismiss' },
  { keys: 'E', label: 'Escalate' },
  { keys: 'U', label: 'Undo' },
]

const SEARCH_FIELDS: Array<{
  key: SearchFieldKey
  label: string
  values: (transaction: TransactionFlag) => string[]
}> = [
  {
    key: 'transaction_id',
    label: 'transaction_id',
    values: (transaction) => [transaction.transactionId],
  },
  {
    key: 'timestamp',
    label: 'timestamp',
    values: (transaction) => {
      const values = [transaction.timestamp]
      const date = new Date(transaction.timestamp)
      if (!Number.isNaN(date.getTime())) {
        values.push(date.toISOString().slice(0, 10))
        values.push(
          date.toLocaleDateString(undefined, {
            day: '2-digit',
            month: 'short',
            year: 'numeric',
          }),
        )
      }

      return values
    },
  },
  {
    key: 'card_id',
    label: 'card_id',
    values: (transaction) => [transaction.cardId],
  },
  {
    key: 'amount',
    label: 'amount',
    values: (transaction) => [
      String(transaction.amount),
      transaction.amount.toFixed(2),
      `$${transaction.amount.toFixed(2)}`,
    ],
  },
  {
    key: 'merchant_name',
    label: 'merchant_name',
    values: (transaction) => [transaction.merchantName],
  },
  {
    key: 'merchant_category',
    label: 'merchant_category',
    values: (transaction) => [transaction.merchantCategory],
  },
  {
    key: 'channel',
    label: 'channel',
    values: (transaction) => [transaction.channel],
  },
  {
    key: 'cardholder_country',
    label: 'cardholder_country',
    values: (transaction) => [transaction.cardholderCountry],
  },
  {
    key: 'merchant_country',
    label: 'merchant_country',
    values: (transaction) => [transaction.merchantCountry],
  },
  {
    key: 'device_id',
    label: 'device_id',
    values: (transaction) => [transaction.deviceId ?? ''],
  },
  {
    key: 'ip_address',
    label: 'ip_address',
    values: (transaction) => [transaction.ipAddress ?? ''],
  },
]

const SEARCH_FIELD_MAP = new Map(SEARCH_FIELDS.map((field) => [field.key, field]))

export function ReviewQueue({
  activeFileHash,
  fileHash,
  items,
  onReset,
  onSelectSession,
  sessions,
  useModel,
}: ReviewQueueProps) {
  const [transactions, setTransactions] = useState(items)
  const [activeId, setActiveId] = useState(items[0]?.transactionId ?? '')
  const [filter, setFilter] = useState<QueueFilter>('pending')
  const [query, setQuery] = useState('')
  const [searchMode, setSearchMode] = useState<SearchMode>('all')
  const [singleField, setSingleField] = useState<SearchFieldKey>('transaction_id')
  const [customFields, setCustomFields] = useState<SearchFieldKey[]>([
    'transaction_id',
    'card_id',
    'merchant_name',
  ])
  const [shortcutsOpen, setShortcutsOpen] = useState(false)
  const [history, setHistory] = useState<DecisionAction[]>([])
  const [networkFocus, setNetworkFocus] = useState<{
    label: string
    transactionIds: Set<string>
  } | null>(null)
  const {
    isError: reviewSyncFailed,
    mutate: syncReviewDecision,
  } = useMutation({
    mutationFn: submitReviewDecision,
  })

  const searchScopeKeys = useMemo(() => {
    if (searchMode === 'single') {
      return [singleField]
    }

    if (searchMode === 'custom') {
      return customFields.length > 0
        ? customFields
        : SEARCH_FIELDS.map((field) => field.key)
    }

    return SEARCH_FIELDS.map((field) => field.key)
  }, [customFields, searchMode, singleField])

  const visibleEntries = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase()

    return transactions
      .map((transaction) => {
        const filterPasses =
          filter === 'all' ? true : transaction.decision === filter
        const networkPasses = networkFocus
          ? networkFocus.transactionIds.has(transaction.transactionId)
          : true

        let matchedFieldLabels: string[] = []

        if (normalizedQuery) {
          matchedFieldLabels = searchScopeKeys
            .flatMap((key) => {
              const field = SEARCH_FIELD_MAP.get(key)
              if (!field) {
                return []
              }

              const hasMatch = field
                .values(transaction)
                .some((value) => value.toLowerCase().includes(normalizedQuery))

              return hasMatch ? [field.label] : []
            })
            .filter((value, index, array) => array.indexOf(value) === index)
        }

        const queryPasses =
          !normalizedQuery || matchedFieldLabels.length > 0

        return {
          matchedFieldLabels,
          transaction,
          visible: filterPasses && networkPasses && queryPasses,
        }
      })
      .filter((entry) => entry.visible)
  }, [filter, networkFocus, query, searchScopeKeys, transactions])

  const visibleTransactions = useMemo(
    () => visibleEntries.map((entry) => entry.transaction),
    [visibleEntries],
  )

  const matchFieldsByTransactionId = useMemo(
    () =>
      new Map(
        visibleEntries.map((entry) => [
          entry.transaction.transactionId,
          entry.matchedFieldLabels,
        ]),
      ),
    [visibleEntries],
  )

  const activeTransaction =
    visibleTransactions.find((transaction) => transaction.transactionId === activeId) ??
    visibleTransactions[0]

  const reviewableTransactionIds = useMemo(
    () => new Set(transactions.map((transaction) => transaction.transactionId)),
    [transactions],
  )

  const activeCardAnalysisQuery = useQuery({
    enabled: Boolean(activeTransaction?.cardId),
    queryFn: () =>
      fetchCardAnalysis({
        cardId: activeTransaction?.cardId ?? '',
        fileHash,
        useModel,
      }),
    queryKey: ['card-analysis', fileHash, activeTransaction?.cardId, useModel],
  })

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
    setSearchMode('all')
    setSingleField('transaction_id')
    setCustomFields(['transaction_id', 'card_id', 'merchant_name'])
    setNetworkFocus(null)
    setHistory([])
  }, [fileHash, items])

  const focusRelatedTransactions = useCallback(
    ({ label, transactionIds }: { label: string; transactionIds: string[] }) => {
      if (transactionIds.length === 0) {
        return
      }

      const ids = new Set(transactionIds)
      const firstVisible = transactions.find((tx) => ids.has(tx.transactionId))

      setFilter('all')
      setQuery('')
      setNetworkFocus({ label, transactionIds: ids })
      if (firstVisible) {
        setActiveId(firstVisible.transactionId)
      }
    },
    [transactions],
  )

  const toggleCustomField = (key: SearchFieldKey) => {
    setCustomFields((current) =>
      current.includes(key)
        ? current.filter((value) => value !== key)
        : [...current, key],
    )
  }

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

      const activeIndex = visibleTransactions.findIndex(
        (item) => item.transactionId === transactionId,
      )
      const nextActiveTransaction =
        visibleTransactions[activeIndex + 1] ??
        visibleTransactions[activeIndex - 1] ??
        null
      const action: DecisionAction = {
        actedAt: new Date().toISOString(),
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
      setActiveId(nextActiveTransaction?.transactionId ?? '')
      syncReviewDecision({
        decision: nextDecision,
        fileHash,
        transactionId,
      })
    },
    [fileHash, syncReviewDecision, transactions, visibleTransactions],
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
    syncReviewDecision({
      decision: lastAction.previousDecision,
      fileHash,
      transactionId: lastAction.transactionId,
    })
  }, [fileHash, history, syncReviewDecision])

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

      if (target?.matches('input, textarea, select, button')) {
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
            <span>{transactions.length} in queue</span>
            {networkFocus ? <span>network: {networkFocus.label}</span> : null}
            {reviewSyncFailed ? <span>Sync failed</span> : null}
            <span>{useModel ? 'ML model' : 'Rules model'}</span>
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
                    {session.label} - {session.useModel ? 'ML' : 'Rules'} - {session.fileHash.slice(0, 8)}
                  </option>
                ))}
              </select>
            ) : null}
            <div className="search-toolbar">
              <Input
                aria-label="Search transactions"
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search all columns"
                value={query}
              />
              <label className="search-control compact" htmlFor="search-mode-select">
                <span>Scope</span>
                <select
                  className="search-select"
                  id="search-mode-select"
                  onChange={(event) => setSearchMode(event.target.value as SearchMode)}
                  value={searchMode}
                >
                  <option value="all">All columns</option>
                  <option value="single">One column</option>
                  <option value="custom">Custom set</option>
                </select>
              </label>

              {searchMode === 'single' ? (
                <label className="search-control compact" htmlFor="single-column-select">
                  <span>Column</span>
                  <select
                    className="search-select"
                    id="single-column-select"
                    onChange={(event) =>
                      setSingleField(event.target.value as SearchFieldKey)
                    }
                    value={singleField}
                  >
                    {SEARCH_FIELDS.map((field) => (
                      <option key={field.key} value={field.key}>
                        {field.label}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
            </div>
            <div className="shortcut-menu-wrap">
              <Button
                aria-expanded={shortcutsOpen}
                aria-haspopup="true"
                onClick={() => setShortcutsOpen((open) => !open)}
                size="sm"
                variant="outline"
              >
                Shortcuts
              </Button>
              {shortcutsOpen ? (
                <div className="shortcut-menu" role="menu">
                  {shortcutOptions.map((shortcut) => (
                    <div className="shortcut-menu-row" key={shortcut.label}>
                      <kbd>{shortcut.keys}</kbd>
                      <span>{shortcut.label}</span>
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
            <Button
              disabled={history.length === 0}
              onClick={undo}
              size="sm"
              variant="outline"
            >
              Undo
            </Button>
            {networkFocus ? (
              <Button
                onClick={() => setNetworkFocus(null)}
                size="sm"
                variant="outline"
              >
                Clear Network Filter
              </Button>
            ) : null}
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

          {searchMode === 'custom' ? (
            <div className="search-custom-fields" aria-label="Custom search columns">
              {SEARCH_FIELDS.map((field) => {
                const selected = customFields.includes(field.key)

                return (
                  <button
                    className={selected ? 'search-chip search-chip-active' : 'search-chip'}
                    key={field.key}
                    onClick={() => toggleCustomField(field.key)}
                    type="button"
                  >
                    {field.label}
                  </button>
                )
              })}
            </div>
          ) : null}
        </div>

        <div className="review-layout">
          <QueueList
            activeTransactionId={activeTransaction?.transactionId}
            matchFieldsByTransactionId={matchFieldsByTransactionId}
            onSelect={setActiveId}
            searchQuery={query}
            transactions={visibleTransactions}
          />

          {activeTransaction ? (
            <TransactionDetail
              cardAnalysis={activeCardAnalysisQuery.data ?? null}
              cardAnalysisError={
                activeCardAnalysisQuery.error instanceof Error
                  ? activeCardAnalysisQuery.error.message
                  : null
              }
              isCardAnalysisLoading={activeCardAnalysisQuery.isFetching}
              onDecide={decide}
              onFocusRelatedTransactions={focusRelatedTransactions}
              onSelectTransaction={setActiveId}
              reviewableTransactionIds={reviewableTransactionIds}
              transactions={transactions}
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
