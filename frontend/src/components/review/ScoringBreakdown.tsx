import { formatPercent, type MlScoringInfo } from '../../lib/mlScoring'
import type { TransactionFlag } from '../../types'

type ScoringBreakdownProps = {
  detailUseModel: boolean
  heuristicTransaction: TransactionFlag
  isModelLoading?: boolean
  mlModelAvailable: boolean
  modelThreshold: number | null
  modelTransaction: TransactionFlag | null
  onDetailUseModelChange: (useModel: boolean) => void
}

export function ScoringBreakdown({
  detailUseModel,
  heuristicTransaction,
  isModelLoading = false,
  mlModelAvailable,
  modelThreshold,
  modelTransaction,
  onDetailUseModelChange,
}: ScoringBreakdownProps) {
  const showModelView = detailUseModel && mlModelAvailable
  const transaction =
    showModelView && modelTransaction ? modelTransaction : heuristicTransaction

  return (
    <section aria-label="Scoring" className="scoring-breakdown">
      <div className="scoring-breakdown-header">
        <h3>Scoring</h3>
        {mlModelAvailable ? (
          <DetailScorerToggle
            disabled={isModelLoading && !modelTransaction}
            onChange={onDetailUseModelChange}
            useModel={detailUseModel}
          />
        ) : null}
      </div>

      {showModelView && isModelLoading && !modelTransaction ? (
        <p className="scoring-breakdown-note">Loading ML scoring for this transaction…</p>
      ) : showModelView ? (
        <ModelScoringContent
          modelThreshold={modelThreshold}
          transaction={transaction}
        />
      ) : (
        <HeuristicScoringContent transaction={transaction} />
      )}
    </section>
  )
}

function DetailScorerToggle({
  disabled,
  onChange,
  useModel,
}: {
  disabled?: boolean
  onChange: (useModel: boolean) => void
  useModel: boolean
}) {
  return (
    <div
      aria-label="Compare scoring for this transaction"
      className="scoring-toggle scoring-toggle-detail"
      role="group"
    >
      <span
        className={
          useModel
            ? 'scoring-toggle-label'
            : 'scoring-toggle-label scoring-toggle-label-active'
        }
      >
        Heuristic
      </span>
      <label className="scoring-switch">
        <input
          checked={useModel}
          className="scoring-switch-input"
          disabled={disabled}
          onChange={(event) => onChange(event.target.checked)}
          type="checkbox"
        />
        <span className="scoring-switch-track">
          <span className="scoring-switch-thumb" />
        </span>
      </label>
      <span
        className={
          useModel
            ? 'scoring-toggle-label scoring-toggle-label-active'
            : 'scoring-toggle-label'
        }
      >
        ML model
      </span>
    </div>
  )
}

function HeuristicScoringContent({
  transaction,
}: {
  transaction: TransactionFlag
}) {
  return (
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
  )
}

function ModelScoringContent({
  modelThreshold,
  transaction,
}: {
  modelThreshold: number | null
  transaction: TransactionFlag
}) {
  const ml: MlScoringInfo | undefined = transaction.mlScoring
  const threshold = ml?.modelThreshold ?? modelThreshold

  return (
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
