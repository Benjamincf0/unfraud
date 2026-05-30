import { formatCurrency } from '../../lib/utils'
import type { TransactionFlag } from '../../types'

type QueueListProps = {
  activeTransactionId?: string
  onSelect: (transactionId: string) => void
  transactions: TransactionFlag[]
}

export function QueueList({
  activeTransactionId,
  onSelect,
  transactions,
}: QueueListProps) {
  return (
    <section className="queue-list" aria-label="Transaction queue">
      <div className="queue-list-body">
        {transactions.length === 0 ? (
          <p className="empty-copy">No transactions match the current view.</p>
        ) : (
          transactions.map((transaction) => (
            <button
              className={
                transaction.transactionId === activeTransactionId
                  ? 'queue-item queue-item-active'
                  : 'queue-item'
              }
              key={transaction.transactionId}
              onClick={() => onSelect(transaction.transactionId)}
              type="button"
            >
              <span>
                <strong>{transaction.merchantName}</strong>
                <span className="queue-item-id">{transaction.transactionId}</span>
              </span>
              <span className="queue-item-meta">
                {formatCurrency(transaction.amount)}
                <span>{Math.round(transaction.score * 100)} risk</span>
              </span>
            </button>
          ))
        )}
      </div>
    </section>
  )
}
