import type {
  ReviewSummary,
  ReviewSessionData,
} from '../lib/scoringViews'
import { QUEUE_BOOTSTRAP_LIMIT } from '../lib/scoringViews'
import type {
  CardAnalysis,
  ReviewLogEntry,
  ReviewDecision,
  RiskReason,
  TransactionFlag,
} from '../types'

export type { ReviewSessionData, ReviewSummary }

type BackendUploadResponse = {
  file_hash: string
  message: string
}

type BackendSummaryResponse = {
  total_transactions: number
  flagged_count: number
  ml_model_available: boolean
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
}

type BackendScorerDetail = {
  fraud_score: number
  is_fraud: boolean
  reasons?: string[]
  score_breakdown?: BackendRiskSignal[]
  card_baseline?: BackendCardBaseline
  cross_card_signals?: Record<string, unknown>
  graph_features?: Record<string, number>
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
  reviewed_at: string
}

export type TransactionDetailResult = {
  transactionId: string
  heuristic: BackendScorerDetail
  model: BackendScorerDetail | null
}

export type ReviewBootstrapResult = {
  heuristic: TransactionFlag[]
  model: TransactionFlag[]
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
    transactionId,
    useModel = false,
  }: {
    flaggedOnly?: boolean
    limit?: number
    offset?: number
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
    mlModelAvailable: payload.ml_model_available,
  }
}

export async function fetchReviewQueuePage(
  fileHash: string,
  {
    flaggedOnly = true,
    limit,
    offset = 0,
    transactionId,
    useModel = false,
  }: {
    flaggedOnly?: boolean
    limit?: number
    offset?: number
    transactionId?: string
    useModel?: boolean
  } = {},
): Promise<{ items: TransactionFlag[]; total: number }> {
  const response = await fetch(
    buildQueueUrl(fileHash, {
      flaggedOnly,
      limit,
      offset,
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

export async function fetchReviewBootstrap(
  fileHash: string,
  summary: ReviewSummary,
): Promise<ReviewBootstrapResult> {
  const heuristicPage = await fetchReviewQueuePage(fileHash, {
    limit: QUEUE_BOOTSTRAP_LIMIT,
    offset: 0,
    useModel: false,
  })

  if (!summary.mlModelAvailable) {
    return {
      heuristic: heuristicPage.items,
      model: [],
    }
  }

  try {
    const modelPage = await fetchReviewQueuePage(fileHash, {
      limit: QUEUE_BOOTSTRAP_LIMIT,
      offset: 0,
      useModel: true,
    })

    return {
      heuristic: heuristicPage.items,
      model: modelPage.items,
    }
  } catch {
    return {
      heuristic: heuristicPage.items,
      model: [],
    }
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
  fileHash,
  transactionId,
}: {
  decision: ReviewDecision
  fileHash: string
  transactionId: string
}) {
  const action = decisionToBackendAction(decision)
  const response = await fetch(
    `${apiBaseUrl}/review/${fileHash}/${transactionId}/${action}`,
    {
      body: JSON.stringify({ action }),
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
      reviewerNotes: item.reviewer_notes ?? undefined,
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
  return toTransactionFlag(
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
}

export function applyScorerDetailToTransaction(
  transaction: TransactionFlag,
  detail: BackendScorerDetail,
  transactionId: string,
): TransactionFlag {
  return {
    ...transaction,
    score: detail.fraud_score,
    label: getRiskLabel(detail.fraud_score),
    isFraud: detail.is_fraud,
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
