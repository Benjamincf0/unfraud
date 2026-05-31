import type {
  FlaggedQueueStats,
  ReviewSummary,
  ReviewSessionData,
} from '../lib/scoringViews'
import { emptyFlaggedQueueStats } from '../lib/scoringViews'
import { QUEUE_FETCH_PAGE_SIZE } from '../lib/scoringViews'
import { mlScoringFromBackend } from '../lib/mlScoring'
import type {
  CardAnalysis,
  ReviewLogEntry,
  ReviewDecision,
  RiskReason,
  TransactionFlag,
  DecisionFeedback,
} from '../types'

export type { ReviewSessionData, ReviewSummary }

type BackendUploadResponse = {
  file_hash: string
  message: string
}

type BackendFlaggedQueueStats = {
  pending: number
  approved: number
  dismissed: number
  escalated: number
}

type BackendSummaryResponse = {
  total_transactions: number
  flagged_count: number
  model_flagged_count?: number
  ml_model_available: boolean
  flagged_queue_stats?: BackendFlaggedQueueStats
  model_flagged_queue_stats?: BackendFlaggedQueueStats
  model_threshold?: number | null
  model_only_count?: number
  alert_only_count?: number
  model_alert_both_count?: number
  soft_rule_only_count?: number
}

type BackendQueuePageResponse = {
  items: BackendQueueItem[]
  total: number
  offset: number
  limit?: number | null
}

type BackendQueueItem = {
  transaction_id: string
  timestamp: string
  card_id: string
  amount: number
  merchant_name: string
  merchant_category: string
  channel: 'online' | 'in_person' | 'atm'
  cardholder_country: string
  merchant_country: string
  device_id?: string | null
  ip_address?: string | null
  is_fraud: boolean
  fraud_score: number
  fraud_reasons?: string[]
  review_decision?: string
  reviewed_at?: string | null
  reviewer_notes?: string | null
  card_baseline?: BackendCardBaseline
  model_score?: number | null
  flagged_by_model?: boolean | null
  flagged_by_alert?: boolean | null
  rule_guardrail?: boolean | null
}

type BackendScorerDetail = {
  fraud_score: number
  is_fraud: boolean
  reasons?: string[]
  score_breakdown?: BackendRiskSignal[]
  card_baseline?: BackendCardBaseline
  cross_card_signals?: Record<string, unknown>
  graph_features?: Record<string, number>
  model_score?: number | null
  model_threshold?: number | null
  flagged_by_model?: boolean | null
  flagged_by_alert?: boolean | null
  rule_guardrail?: boolean | null
}

type BackendTransactionDetailResponse = BackendQueueItem & {
  heuristic: BackendScorerDetail
  model?: BackendScorerDetail | null
}

type BackendRiskSignal = {
  code?: string
  label?: string
  detail?: string
  weight?: number
  signal_type?: string
}

type BackendCardBaseline = {
  history_count?: number
  typical_amount?: number
  usual_categories?: string[]
  usual_countries?: string[]
}

type BackendAnalyzedTransaction = {
  transaction_id: string
  timestamp: string
  card_id: string
  amount: number
  merchant_name: string
  merchant_category: string
  channel: 'online' | 'in_person' | 'atm'
  cardholder_country: string
  merchant_country: string
  device_id?: string
  ip_address?: string
  fraud_reasons?: string[]
  fraud_score: number
  is_fraud: boolean
  reasons?: string[]
  score_breakdown?: BackendRiskSignal[]
}

type BackendReviewLogEntry = {
  transaction_id: string
  action: 'approve' | 'dismiss' | 'escalate'
  reviewer_notes?: string | null
  feedback_reason_codes?: string[]
  feedback_reasoning?: string | null
  feedback_effects?: BackendReviewFeedbackEffect[]
  reviewed_at: string
}

type BackendReviewFeedbackEffect = {
  type: string
  signal_code: string
  signal_label: string
  direction: string
  previous_multiplier: number
  next_multiplier: number
  summary: string
}

export type TransactionDetailResult = {
  transactionId: string
  heuristic: BackendScorerDetail
  model: BackendScorerDetail | null
}

export type ReviewQueueLoadResult = {
  heuristic: TransactionFlag[]
  model: TransactionFlag[]
}

export type ReviewQueueLoadProgress = {
  heuristicLoaded: number
  heuristicTotal: number
  modelLoaded: number
  modelTotal: number
}

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? '/api'

function withUseModel(url: string, useModel: boolean) {
  if (!useModel) {
    return url
  }

  const separator = url.includes('?') ? '&' : '?'
  return `${url}${separator}use_model=true`
}

function buildQueueUrl(
  fileHash: string,
  {
    flaggedOnly = true,
    limit,
    offset = 0,
    slim = false,
    transactionId,
    useModel = false,
  }: {
    flaggedOnly?: boolean
    limit?: number
    offset?: number
    slim?: boolean
    transactionId?: string
    useModel?: boolean
  } = {},
) {
  const params = new URLSearchParams()

  if (useModel) {
    params.set('use_model', 'true')
  }

  if (!flaggedOnly) {
    params.set('flagged_only', 'false')
  }

  if (slim) {
    params.set('slim', 'true')
  }

  if (typeof limit === 'number') {
    params.set('limit', String(limit))
  }

  if (offset > 0) {
    params.set('offset', String(offset))
  }

  if (transactionId) {
    params.set('transaction_id', transactionId)
  }

  const query = params.toString()
  return `${apiBaseUrl}/analysis/queue/${fileHash}${query ? `?${query}` : ''}`
}

export async function fetchReviewSummary(fileHash: string): Promise<ReviewSummary> {
  const response = await fetch(`${apiBaseUrl}/analysis/summary/${fileHash}`)

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Summary failed'))
  }

  const payload = (await response.json()) as BackendSummaryResponse

  return {
    totalTransactions: payload.total_transactions,
    flaggedCount: payload.flagged_count,
    modelFlaggedCount: payload.model_flagged_count ?? 0,
    mlModelAvailable: payload.ml_model_available,
    flaggedQueueStats: mapFlaggedQueueStats(
      payload.flagged_queue_stats,
      payload.flagged_count,
    ),
    modelFlaggedQueueStats: mapFlaggedQueueStats(
      payload.model_flagged_queue_stats,
      payload.model_flagged_count ?? 0,
    ),
    modelThreshold: payload.model_threshold ?? null,
    modelOnlyCount: payload.model_only_count ?? 0,
    alertOnlyCount: payload.alert_only_count ?? 0,
    modelAlertBothCount: payload.model_alert_both_count ?? 0,
    softRuleOnlyCount: payload.soft_rule_only_count ?? 0,
  }
}

export async function fetchReviewQueuePage(
  fileHash: string,
  {
    flaggedOnly = true,
    limit,
    offset = 0,
    slim = false,
    transactionId,
    useModel = false,
  }: {
    flaggedOnly?: boolean
    limit?: number
    offset?: number
    slim?: boolean
    transactionId?: string
    useModel?: boolean
  } = {},
): Promise<{ items: TransactionFlag[]; total: number }> {
  const response = await fetch(
    buildQueueUrl(fileHash, {
      flaggedOnly,
      limit,
      offset,
      slim,
      transactionId,
      useModel,
    }),
  )

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Queue fetch failed'))
  }

  const payload = (await response.json()) as BackendQueuePageResponse

  return {
    items: payload.items.map((item) => mapQueueItemToTransactionFlag(item)),
    total: payload.total,
  }
}

export async function fetchAllReviewQueue(
  fileHash: string,
  {
    flaggedOnly = true,
    onChunk,
    onProgress,
    useModel = false,
  }: {
    flaggedOnly?: boolean
    onChunk?: (items: TransactionFlag[]) => void
    onProgress?: (loaded: number, total: number) => void
    useModel?: boolean
  } = {},
): Promise<TransactionFlag[]> {
  const firstPage = await fetchReviewQueuePage(fileHash, {
    flaggedOnly,
    limit: QUEUE_FETCH_PAGE_SIZE,
    offset: 0,
    slim: true,
    useModel,
  })

  const items = [...firstPage.items]
  onChunk?.(firstPage.items)
  onProgress?.(items.length, firstPage.total)

  let offset = items.length
  while (offset < firstPage.total) {
    const page = await fetchReviewQueuePage(fileHash, {
      flaggedOnly,
      limit: QUEUE_FETCH_PAGE_SIZE,
      offset,
      slim: true,
      useModel,
    })

    items.push(...page.items)
    onChunk?.(page.items)
    offset += page.items.length
    onProgress?.(offset, firstPage.total)
  }

  return items
}

export async function fetchFullReviewQueue(
  fileHash: string,
  summary: ReviewSummary,
  {
    onHeuristicChunk,
    onModelChunk,
    onProgress,
  }: {
    onHeuristicChunk?: (items: TransactionFlag[]) => void
    onModelChunk?: (items: TransactionFlag[]) => void
    onProgress?: (progress: ReviewQueueLoadProgress) => void
  } = {},
): Promise<ReviewQueueLoadResult> {
  const heuristic = await fetchAllReviewQueue(fileHash, {
    onChunk: onHeuristicChunk,
    onProgress: (loaded, total) => {
      onProgress?.({
        heuristicLoaded: loaded,
        heuristicTotal: total,
        modelLoaded: 0,
        modelTotal: summary.modelFlaggedCount,
      })
    },
  })

  if (!summary.mlModelAvailable) {
    onProgress?.({
      heuristicLoaded: heuristic.length,
      heuristicTotal: heuristic.length,
      modelLoaded: 0,
      modelTotal: 0,
    })

    return { heuristic, model: [] }
  }

  try {
    const model = await fetchAllReviewQueue(fileHash, {
      onChunk: onModelChunk,
      onProgress: (loaded, total) => {
        onProgress?.({
          heuristicLoaded: heuristic.length,
          heuristicTotal: heuristic.length,
          modelLoaded: loaded,
          modelTotal: total,
        })
      },
      useModel: true,
    })

    return { heuristic, model }
  } catch {
    onProgress?.({
      heuristicLoaded: heuristic.length,
      heuristicTotal: heuristic.length,
      modelLoaded: 0,
      modelTotal: summary.modelFlaggedCount,
    })

    return { heuristic, model: [] }
  }
}

export async function fetchTransactionDetail(
  fileHash: string,
  transactionId: string,
): Promise<TransactionDetailResult> {
  const response = await fetch(
    `${apiBaseUrl}/analysis/transaction/${fileHash}/${encodeURIComponent(transactionId)}`,
  )

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Transaction detail failed'))
  }

  const payload = (await response.json()) as BackendTransactionDetailResponse

  return {
    transactionId: payload.transaction_id,
    heuristic: payload.heuristic,
    model: payload.model ?? null,
  }
}

export async function fetchRelatedTransactions(
  fileHash: string,
  transactionId: string,
  useModel = false,
): Promise<TransactionFlag[]> {
  const response = await fetch(
    withUseModel(
      `${apiBaseUrl}/analysis/related/${fileHash}/${encodeURIComponent(transactionId)}`,
      useModel,
    ),
  )

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Related transactions failed'))
  }

  const payload = (await response.json()) as { items: BackendQueueItem[] }

  return payload.items.map((item) => mapQueueItemToTransactionFlag(item))
}

export async function openReviewSession(fileHash: string): Promise<ReviewSessionData> {
  const summary = await fetchReviewSummary(fileHash)

  return {
    fileHash,
    summary,
    source: 'cache',
  }
}

export async function uploadTransactionsCsv(file: File): Promise<ReviewSessionData> {
  const formData = new FormData()
  formData.append('file', file)

  const uploadResponse = await fetch(`${apiBaseUrl}/upload`, {
    body: formData,
    method: 'POST',
  })

  if (!uploadResponse.ok) {
    throw new Error(await getErrorMessage(uploadResponse, 'Upload failed'))
  }

  const uploadPayload = (await uploadResponse.json()) as BackendUploadResponse

  if (!uploadPayload.file_hash) {
    throw new Error('Upload returned an invalid file hash')
  }

  const summary = await fetchReviewSummary(uploadPayload.file_hash)

  return {
    fileHash: uploadPayload.file_hash,
    summary,
    source: 'upload',
  }
}

export async function submitReviewDecision({
  decision,
  feedback,
  fileHash,
  transactionId,
}: {
  decision: ReviewDecision
  feedback?: DecisionFeedback
  fileHash: string
  transactionId: string
}) {
  const action = decisionToBackendAction(decision)
  const feedbackReasoning = feedback?.reasoning.trim()
  const response = await fetch(
    `${apiBaseUrl}/review/${fileHash}/${transactionId}/${action}`,
    {
      body: JSON.stringify({
        action,
        feedback_reason_codes: feedback?.reasonCodes ?? [],
        feedback_reasoning: feedbackReasoning || undefined,
        reviewer_notes: feedbackReasoning || undefined,
      }),
      headers: {
        'Content-Type': 'application/json',
      },
      method: 'POST',
    },
  )

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Review update failed'))
  }

  return response.json()
}

export async function fetchReviewLog(
  fileHash: string,
): Promise<ReviewLogEntry[]> {
  const response = await fetch(`${apiBaseUrl}/review-log/${fileHash}`)

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Review log failed'))
  }

  const payload = (await response.json()) as BackendReviewLogEntry[]
  if (!Array.isArray(payload)) {
    throw new Error('Review log returned an invalid response')
  }

  return payload
    .filter((item) => item && item.transaction_id && item.action)
    .map((item) => ({
      transactionId: item.transaction_id,
      action: backendActionToDecision(item.action),
      reviewerNotes:
        item.feedback_reasoning ?? item.reviewer_notes ?? undefined,
      feedbackEffects: (item.feedback_effects ?? []).map((effect) => ({
        direction: effect.direction,
        nextMultiplier: effect.next_multiplier,
        previousMultiplier: effect.previous_multiplier,
        signalCode: effect.signal_code,
        signalLabel: effect.signal_label,
        summary: effect.summary,
        type: effect.type,
      })),
      reviewedAt: item.reviewed_at,
    }))
}

export async function fetchCardAnalysis({
  cardId,
  fileHash,
  useModel = false,
}: {
  cardId: string
  fileHash: string
  useModel?: boolean
}): Promise<CardAnalysis> {
  const response = await fetch(
    withUseModel(
      `${apiBaseUrl}/analysis/user/${fileHash}/${encodeURIComponent(cardId)}`,
      useModel,
    ),
  )

  if (!response.ok) {
    throw new Error(await getErrorMessage(response, 'Card analysis failed'))
  }

  const payload = await response.json()

  if (!Array.isArray(payload)) {
    throw new Error('Card analysis returned an invalid response')
  }

  return toCardAnalysis(cardId, payload.map(normalizeCardTransactionPayload))
}

export function mapQueueItemToTransactionFlag(item: BackendQueueItem): TransactionFlag {
  const flag = toTransactionFlag(
    item,
    {
      fraud_score: item.fraud_score,
      is_fraud: item.is_fraud,
      reasons: item.fraud_reasons ?? [],
      score_breakdown: [],
      transaction_id: item.transaction_id,
      card_baseline: item.card_baseline,
    },
    undefined,
    parseReviewDecision(item.review_decision ?? ''),
    emptyToUndefined(item.reviewed_at ?? undefined),
    emptyToUndefined(item.reviewer_notes ?? undefined),
  )
  const mlScoring = mlScoringFromBackend(item)
  return mlScoring ? { ...flag, mlScoring } : flag
}

export function applyScorerDetailToTransaction(
  transaction: TransactionFlag,
  detail: BackendScorerDetail,
  transactionId: string,
): TransactionFlag {
  const mlScoring = mlScoringFromBackend(detail)
  return {
    ...transaction,
    score: detail.fraud_score,
    label: getRiskLabel(detail.fraud_score),
    isFraud: detail.is_fraud,
    mlScoring: mlScoring ?? transaction.mlScoring,
    reasons: buildReasons({
      fraud_score: detail.fraud_score,
      is_fraud: detail.is_fraud,
      reasons: detail.reasons ?? [],
      score_breakdown: detail.score_breakdown,
      transaction_id: transactionId,
    }),
    cardContext: detail.card_baseline
      ? {
          medianAmount:
            typeof detail.card_baseline.typical_amount === 'number'
              ? detail.card_baseline.typical_amount
              : transaction.cardContext.medianAmount,
          previousTransactions:
            typeof detail.card_baseline.history_count === 'number'
              ? detail.card_baseline.history_count
              : transaction.cardContext.previousTransactions,
          usualCategories:
            detail.card_baseline.usual_categories ??
            transaction.cardContext.usualCategories,
          usualCountries:
            detail.card_baseline.usual_countries ??
            transaction.cardContext.usualCountries,
        }
      : transaction.cardContext,
  }
}

export async function fetchExplorerDataset(
  fileHash: string,
  {
    onChunk,
    onProgress,
    useModel = false,
  }: {
    onChunk?: (items: TransactionFlag[]) => void
    onProgress?: (loaded: number, total: number) => void
    useModel?: boolean
  } = {},
): Promise<TransactionFlag[]> {
  return fetchAllReviewQueue(fileHash, {
    flaggedOnly: false,
    onChunk,
    onProgress,
    useModel,
  })
}

function toCardAnalysis(
  cardId: string,
  transactions: BackendAnalyzedTransaction[],
): CardAnalysis {
  const sortedTransactions = [...transactions].sort(
    (first, second) =>
      new Date(first.timestamp).getTime() - new Date(second.timestamp).getTime(),
  )
  const amounts = sortedTransactions
    .map((transaction) => transaction.amount)
    .sort((first, second) => first - second)
  const mid = Math.floor(amounts.length / 2)
  const medianAmount =
    amounts.length === 0
      ? 0
      : amounts.length % 2 === 0
      ? (amounts[mid - 1] + amounts[mid]) / 2
      : amounts[mid]

  return {
    cardId,
    summary: {
      categories: getMostCommon(
        sortedTransactions.map((transaction) => transaction.merchant_category),
        6,
      ),
      countries: getMostCommon(
        sortedTransactions.map((transaction) => transaction.merchant_country),
        6,
      ),
      fraudCount: sortedTransactions.filter((transaction) => transaction.is_fraud)
        .length,
      highestAmount: Math.max(
        0,
        ...sortedTransactions.map((transaction) => transaction.amount),
      ),
      medianAmount,
      totalSpend: sortedTransactions.reduce(
        (total, transaction) => total + transaction.amount,
        0,
      ),
      transactionCount: sortedTransactions.length,
    },
    transactions: sortedTransactions.map((transaction) => ({
      amount: transaction.amount,
      cardholderCountry: transaction.cardholder_country,
      channel: transaction.channel,
      deviceId: transaction.device_id,
      ipAddress: transaction.ip_address,
      isFraud: transaction.is_fraud,
      merchantCategory: transaction.merchant_category,
      merchantCountry: transaction.merchant_country,
      merchantName: transaction.merchant_name,
      reasons: buildReasons({
        fraud_score: transaction.fraud_score,
        is_fraud: transaction.is_fraud,
        reasons: getBackendReasons(transaction),
        score_breakdown: transaction.score_breakdown,
        transaction_id: transaction.transaction_id,
      }),
      score: transaction.fraud_score,
      timestamp: transaction.timestamp,
      transactionId: transaction.transaction_id,
    })),
  }
}

function normalizeCardTransactionPayload(
  item: unknown,
): BackendAnalyzedTransaction {
  if (!isRecord(item)) {
    throw new Error('Card analysis returned an invalid transaction')
  }

  const transactionId = readString(item, 'transaction_id')
  const amount = readNumber(item, 'amount')

  if (!transactionId || !readString(item, 'timestamp') || amount === null) {
    throw new Error(
      'Card analysis response is missing transaction history. Restart the backend and try again.',
    )
  }

  return {
    amount,
    card_id: readString(item, 'card_id'),
    cardholder_country: readString(item, 'cardholder_country'),
    channel: normalizeChannel(readString(item, 'channel')),
    device_id: emptyToUndefined(readString(item, 'device_id')),
    fraud_reasons: readStringArray(item, 'fraud_reasons'),
    fraud_score: readNumber(item, 'fraud_score') ?? 0,
    ip_address: emptyToUndefined(readString(item, 'ip_address')),
    is_fraud: Boolean(item.is_fraud),
    merchant_category: readString(item, 'merchant_category'),
    merchant_country: readString(item, 'merchant_country'),
    merchant_name: readString(item, 'merchant_name'),
    reasons: readStringArray(item, 'reasons'),
    score_breakdown: readRiskSignals(item, 'score_breakdown'),
    timestamp: readString(item, 'timestamp'),
    transaction_id: transactionId,
  }
}

function getBackendReasons(transaction: BackendAnalyzedTransaction) {
  return transaction.fraud_reasons ?? transaction.reasons ?? []
}

function toTransactionFlag(
  transaction: BackendQueueItem,
  analysis: {
    fraud_score: number
    is_fraud: boolean
    reasons: string[]
    score_breakdown?: BackendRiskSignal[]
    transaction_id: string
    card_baseline?: BackendCardBaseline
  },
  cardContext = {
    medianAmount: 0,
    previousTransactions: 0,
    usualCategories: [] as string[],
    usualCountries: [] as string[],
  },
  decision: ReviewDecision = 'pending',
  reviewedAt?: string,
  reviewerNotes?: string,
): TransactionFlag {
  const backendContext = analysis.card_baseline
    ? {
        medianAmount:
          typeof analysis.card_baseline.typical_amount === 'number'
            ? analysis.card_baseline.typical_amount
            : cardContext.medianAmount,
        previousTransactions:
          typeof analysis.card_baseline.history_count === 'number'
            ? analysis.card_baseline.history_count
            : cardContext.previousTransactions,
        usualCategories:
          analysis.card_baseline.usual_categories ?? cardContext.usualCategories,
        usualCountries:
          analysis.card_baseline.usual_countries ?? cardContext.usualCountries,
      }
    : cardContext

  return {
    amount: transaction.amount,
    cardContext: backendContext,
    cardId: transaction.card_id,
    cardholderCountry: transaction.cardholder_country,
    channel: transaction.channel,
    decision,
    deviceId: emptyToUndefined(transaction.device_id ?? undefined),
    ipAddress: emptyToUndefined(transaction.ip_address ?? undefined),
    isFraud: analysis.is_fraud,
    label: getRiskLabel(analysis.fraud_score),
    merchantCategory: transaction.merchant_category,
    merchantCountry: transaction.merchant_country,
    merchantName: transaction.merchant_name,
    reasons: buildReasons(analysis),
    reviewedAt,
    reviewerNotes,
    score: analysis.fraud_score,
    timestamp: transaction.timestamp,
    transactionId: transaction.transaction_id,
  }
}

function backendActionToDecision(
  action: BackendReviewLogEntry['action'],
): Exclude<ReviewDecision, 'pending'> {
  if (action === 'approve') {
    return 'approved'
  }
  if (action === 'dismiss') {
    return 'dismissed'
  }
  return 'escalated'
}

function getMostCommon(values: string[], limit = 4) {
  const counts = new Map<string, number>()

  for (const value of values.filter(Boolean)) {
    counts.set(value, (counts.get(value) ?? 0) + 1)
  }

  return Array.from(counts.entries())
    .sort((first, second) => second[1] - first[1] || first[0].localeCompare(second[0]))
    .slice(0, limit)
    .map(([value]) => value)
}

function buildReasons(analysis: {
  fraud_score: number
  is_fraud: boolean
  reasons: string[]
  score_breakdown?: BackendRiskSignal[]
  transaction_id: string
}): RiskReason[] {
  const scoreBreakdown = Array.isArray(analysis.score_breakdown)
    ? analysis.score_breakdown.filter((reason) => reason.label || reason.detail)
    : []

  if (scoreBreakdown.length > 0) {
    return scoreBreakdown.map((reason, index) => ({
      code: reason.code,
      detail: reason.detail ?? 'Backend detector returned this signal.',
      id: `${analysis.transaction_id}-${reason.code ?? index}`,
      label: reason.label ?? reason.code ?? 'Detector signal',
      signalType: reason.signal_type,
      weight: Math.round((reason.weight ?? analysis.fraud_score) * 100),
    }))
  }

  const reasons = Array.isArray(analysis.reasons) ? analysis.reasons : []
  const reasonWeight =
    reasons.length > 0
      ? Math.round((analysis.fraud_score * 100) / reasons.length)
      : Math.round(analysis.fraud_score * 100)

  if (reasons.length === 0) {
    return [
      {
        detail: 'Backend detector returned a positive score without a specific reason.',
        id: `${analysis.transaction_id}-score`,
        label: 'Elevated model score',
        weight: reasonWeight,
      },
    ]
  }

  return reasons.map((reason, index) => ({
    detail: getReasonDetail(reason),
    id: `${analysis.transaction_id}-${index}`,
    label: reason,
    weight: reasonWeight,
  }))
}

function getReasonDetail(reason: string) {
  const normalizedReason = reason.toLowerCase()

  if (normalizedReason.includes('high amount')) {
    return 'Transaction amount is more than 3x the median amount for this card.'
  }

  if (normalizedReason.includes('foreign')) {
    return 'Merchant country differs from the cardholder country.'
  }

  if (normalizedReason.includes('missing device')) {
    return 'Online transaction is missing a device identifier or IP address.'
  }

  return 'Flag returned by the backend fraud detector.'
}

function getRiskLabel(score: number): TransactionFlag['label'] {
  if (score >= 0.85) {
    return 'critical'
  }

  if (score >= 0.65) {
    return 'high'
  }

  if (score >= 0.35) {
    return 'medium'
  }

  return 'low'
}

function decisionToBackendAction(decision: ReviewDecision) {
  if (decision === 'pending') {
    return 'pending'
  }

  if (decision === 'approved') {
    return 'approve'
  }

  if (decision === 'dismissed') {
    return 'dismiss'
  }

  return 'escalate'
}

async function getErrorMessage(response: Response, fallback: string) {
  try {
    const payload = (await response.json()) as { detail?: unknown }

    if (typeof payload.detail === 'string') {
      return payload.detail
    }
  } catch {
    // Fall through to the generic status message.
  }

  return `${fallback} with status ${response.status}`
}

function normalizeChannel(channel: string): BackendQueueItem['channel'] {
  if (channel === 'online' || channel === 'in_person' || channel === 'atm') {
    return channel
  }

  throw new Error(`Invalid channel: ${channel}`)
}

function emptyToUndefined(value: string | undefined) {
  return value && value.trim() !== '' ? value : undefined
}

function mapFlaggedQueueStats(
  stats: BackendFlaggedQueueStats | undefined,
  flaggedCount: number,
): FlaggedQueueStats {
  if (!stats) {
    return {
      ...emptyFlaggedQueueStats,
      pending: flaggedCount,
    }
  }

  return {
    pending: stats.pending,
    approved: stats.approved,
    dismissed: stats.dismissed,
    escalated: stats.escalated,
  }
}

function parseReviewDecision(value: string): ReviewDecision {
  if (value === 'approve' || value === 'approved') {
    return 'approved'
  }

  if (value === 'dismiss' || value === 'dismissed') {
    return 'dismissed'
  }

  if (value === 'escalate' || value === 'escalated') {
    return 'escalated'
  }

  return 'pending'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function readString(record: Record<string, unknown>, key: string) {
  const value = record[key]
  return typeof value === 'string' ? value : ''
}

function readNumber(record: Record<string, unknown>, key: string) {
  const value = record[key]

  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }

  if (typeof value === 'string') {
    const numberValue = Number(value)
    return Number.isFinite(numberValue) ? numberValue : null
  }

  return null
}

function readStringArray(record: Record<string, unknown>, key: string) {
  const value = record[key]

  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === 'string')
  }

  if (typeof value === 'string') {
    return value
      .split(';')
      .map((reason) => reason.trim())
      .filter(Boolean)
  }

  return []
}

function readRiskSignals(record: Record<string, unknown>, key: string) {
  const value = record[key]

  if (Array.isArray(value)) {
    return value.filter(isBackendRiskSignal)
  }

  return undefined
}

function isBackendRiskSignal(value: unknown): value is BackendRiskSignal {
  return isRecord(value)
}
