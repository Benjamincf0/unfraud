import { formatPercent, type MlScoringInfo } from '../../lib/mlScoring'
import type { TransactionFlag } from '../../types'

type ScoringBreakdownProps = {
  modelThreshold: number | null
  transaction: TransactionFlag
  useModel: boolean
}

export function ScoringBreakdown({
  modelThreshold,
  transaction,
  useModel,
}: ScoringBreakdownProps) {
  if (!useModel) {
    return (
      <section aria-label="Scoring" className="scoring-breakdown">
        <h3>Scoring</h3>
        <dl className="scoring-breakdown-grid">
          <div>
            <dt>Combined score</dt>
            <dd>{formatPercent(transaction.score)}</dd>
          </div>
          <div>
            <dt>Review queue</dt>
            <dd>{transaction.isFraud ? 'Yes — flagged for review' : 'No'}</dd>
          </div>
        </dl>
        <p className="scoring-breakdown-note">
          Heuristic scoring uses a weighted rule blend. Toggle ML model to see
          model probability and alert-rule breakdown.
        </p>
      </section>
    )
  }

  const ml: MlScoringInfo | undefined = transaction.mlScoring
  const threshold = ml?.modelThreshold ?? modelThreshold

  return (
    <section aria-label="Scoring" className="scoring-breakdown">
      <h3>Scoring</h3>
      <dl className="scoring-breakdown-grid">
        <div>
          <dt>Model probability</dt>
          <dd>
            {ml?.modelScore != null ? formatPercent(ml.modelScore) : '—'}
            {threshold != null ? (
              <span className="scoring-breakdown-sub">
                {' '}
                (queue bar {formatPercent(threshold)})
              </span>
            ) : null}
          </dd>
        </div>
        <div>
          <dt>Combined score</dt>
          <dd>
            {formatPercent(transaction.score)}
            {ml?.ruleGuardrail && !transaction.isFraud ? (
              <span className="scoring-breakdown-sub"> includes soft-rule boost</span>
            ) : null}
          </dd>
        </div>
        <div>
          <dt>Review queue</dt>
          <dd>{describeQueueStatus(transaction, ml)}</dd>
        </div>
        {ml?.ruleGuardrail ? (
          <div>
            <dt>Soft guardrail</dt>
            <dd>
              Fired (+0.35 to combined score)
              {!transaction.isFraud ? ' — does not auto-queue alone' : ''}
            </dd>
          </div>
        ) : null}
      </dl>
    </section>
  )
}

function describeQueueStatus(
  transaction: TransactionFlag,
  ml: MlScoringInfo | undefined,
) {
  if (!transaction.isFraud) {
    if (ml?.ruleGuardrail) {
      return 'No — elevated score only (soft rule)'
    }
    return 'No'
  }

  if (!ml) {
    return 'Yes'
  }

  const parts: string[] = []
  if (ml.flaggedByModel) {
    parts.push('model ≥ threshold')
  }
  if (ml.flaggedByAlert) {
    parts.push('alert rule')
  }
  if (parts.length === 0) {
    return 'Yes'
  }
  return `Yes — ${parts.join(' + ')}`
}
