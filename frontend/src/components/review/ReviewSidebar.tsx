import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { Slider } from '../ui/slider'
import type { AuditEntry, ReviewDecision } from '../../types'

type ReviewSidebarProps = {
  auditLog: AuditEntry[]
  historyCount: number
  onThresholdChange: (value: number) => void
  onUndo: () => void
  shownCount: number
  stats: Record<ReviewDecision, number>
  threshold: number
}

export function ReviewSidebar({
  auditLog,
  historyCount,
  onThresholdChange,
  onUndo,
  shownCount,
  stats,
  threshold,
}: ReviewSidebarProps) {
  return (
    <aside className="sidebar" aria-label="Review summary">
      <div className="brand">
        <span className="brand-mark">FH</span>
        <div>
          <p className="brand-title">Fraud Hunter</p>
          <p className="brand-subtitle">Reviewer queue</p>
        </div>
      </div>

      <nav className="summary-list" aria-label="Queue status">
        <SummaryRow label="Pending" value={stats.pending} />
        <SummaryRow label="Approved" value={stats.approved} />
        <SummaryRow label="Dismissed" value={stats.dismissed} />
        <SummaryRow label="Escalated" value={stats.escalated} />
      </nav>

      <div className="side-section">
        <label htmlFor="cost-slider">Review threshold</label>
        <Slider
          id="cost-slider"
          max={95}
          min={20}
          onChange={(event) => onThresholdChange(Number(event.target.value))}
          value={threshold}
        />
        <div className="threshold-row">
          <span>{threshold}%</span>
          <span>{shownCount} shown</span>
        </div>
      </div>

      <div className="side-section audit-section">
        <div className="section-title-row">
          <h2>Audit Trail</h2>
          <Button
            aria-label="Undo last action"
            disabled={historyCount === 0}
            onClick={onUndo}
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
