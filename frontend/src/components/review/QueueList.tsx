import { useEffect, useRef } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { formatCurrency } from '../../lib/utils'
import type { RiskSortMode } from '../../lib/scoringViews'
import type { TransactionFlag } from '../../types'

const QUEUE_ITEM_ESTIMATE_PX = 72

type QueueListProps = {
  activeTransactionId?: string
  matchFieldsByTransactionId?: Map<string, string[]>
  onSelect: (transactionId: string) => void
  onSortModeChange: (mode: RiskSortMode) => void
  searchQuery?: string
  sortMode: RiskSortMode
  sortModeDisabled?: boolean
  sortModeOptions: Array<{ value: RiskSortMode; label: string }>
  transactions: TransactionFlag[]
}

export function QueueList({
  activeTransactionId,
  matchFieldsByTransactionId,
  onSelect,
  onSortModeChange,
  searchQuery,
  sortMode,
  sortModeDisabled = false,
  sortModeOptions,
  transactions,
}: QueueListProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  const virtualizer = useVirtualizer({
    count: transactions.length,
    estimateSize: () => QUEUE_ITEM_ESTIMATE_PX,
    getScrollElement: () => scrollRef.current,
    overscan: 10,
  })

  useEffect(() => {
    if (!activeTransactionId) {
      return
    }

    const index = transactions.findIndex(
      (transaction) => transaction.transactionId === activeTransactionId,
    )

    if (index >= 0) {
      virtualizer.scrollToIndex(index, { align: 'auto' })
    }
  }, [activeTransactionId, transactions, virtualizer])

  return (
    <section className="queue-list" aria-label="Transaction queue">
      <div className="queue-list-toolbar">
        <label className="queue-sort-control" htmlFor="queue-sort-mode">
          <span>Order by</span>
          <select
            disabled={sortModeDisabled}
            id="queue-sort-mode"
            onChange={(event) =>
              onSortModeChange(event.target.value as RiskSortMode)
            }
            value={sortMode}
          >
            {sortModeOptions.map((option) => (
              <option
                disabled={option.value === 'model' && sortModeDisabled}
                key={option.value}
                value={option.value}
              >
                {option.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="queue-list-body" ref={scrollRef}>
        {transactions.length === 0 ? (
          <p className="empty-copy">No transactions match the current view.</p>
        ) : (
          <div
            className="queue-list-virtual-spacer"
            style={{ height: virtualizer.getTotalSize() }}
          >
            {virtualizer.getVirtualItems().map((virtualRow) => {
              const transaction = transactions[virtualRow.index]
              const matchedFields =
                matchFieldsByTransactionId?.get(transaction.transactionId) ?? []
              const amountText = formatCurrency(transaction.amount)

              return (
                <button
                  className={
                    transaction.transactionId === activeTransactionId
                      ? 'queue-item queue-item-active'
                      : 'queue-item'
                  }
                  key={transaction.transactionId}
                  onClick={() => onSelect(transaction.transactionId)}
                  style={{
                    height: `${virtualRow.size}px`,
                    left: 0,
                    position: 'absolute',
                    top: 0,
                    transform: `translateY(${virtualRow.start}px)`,
                    width: '100%',
                  }}
                  type="button"
                >
                  <span className="queue-item-main">
                    <strong>{highlightText(transaction.merchantName, searchQuery)}</strong>
                    <span className="queue-item-id">
                      {highlightText(transaction.transactionId, searchQuery)}
                    </span>
                    {matchedFields.length > 0 ? (
                      <span className="queue-item-match" title={matchedFields.join(', ')}>
                        Match: {matchedFields.join(', ')}
                      </span>
                    ) : null}
                  </span>
                  <span className="queue-item-meta">
                    <span>
                      {matchedFields.includes('amount')
                        ? highlightLooseText(amountText, searchQuery)
                        : amountText}
                    </span>
                    <span>{Math.round(transaction.score * 100)} risk</span>
                  </span>
                </button>
              )
            })}
          </div>
        )}
      </div>
    </section>
  )
}

function highlightText(text: string, query?: string) {
  const cleanedQuery = query?.trim()

  if (!cleanedQuery) {
    return text
  }

  const lowerText = text.toLowerCase()
  const lowerQuery = cleanedQuery.toLowerCase()
  const matchIndex = lowerText.indexOf(lowerQuery)

  if (matchIndex < 0) {
    return text
  }

  const before = text.slice(0, matchIndex)
  const match = text.slice(matchIndex, matchIndex + cleanedQuery.length)
  const after = text.slice(matchIndex + cleanedQuery.length)

  return (
    <>
      {before}
      <mark>{match}</mark>
      {after}
    </>
  )
}

function highlightLooseText(text: string, query?: string) {
  const cleanedQuery = normalizeLoose(query ?? '')

  if (!cleanedQuery) {
    return text
  }

  const indexedText = Array.from(text).reduce(
    (result, char, index) => {
      const normalizedChar = normalizeLoose(char)

      if (normalizedChar) {
        result.normalized += normalizedChar
        result.indexes.push(index)
      }

      return result
    },
    { indexes: [] as number[], normalized: '' },
  )
  const matchIndex = indexedText.normalized.indexOf(cleanedQuery)

  if (matchIndex < 0) {
    return highlightText(text, query)
  }

  const originalStart = indexedText.indexes[matchIndex]
  const originalEnd =
    indexedText.indexes[matchIndex + cleanedQuery.length - 1] + 1

  return (
    <>
      {text.slice(0, originalStart)}
      <mark>{text.slice(originalStart, originalEnd)}</mark>
      {text.slice(originalEnd)}
    </>
  )
}

function normalizeLoose(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]/g, '')
}
