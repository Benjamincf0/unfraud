import type {
  CardAnalysis,
  ReviewLogEntry,
  ReviewDecision,
  RiskReason,
  TransactionFlag,
} from '../types'

export type ReviewDataResult = {
  fileHash: string
  items: TransactionFlag[]
  source: 'cache' | 'upload'
}

type BackendUploadResponse = {
  file_hash: string
  message: string
}

type BackendFraudAnalysis = {
  transaction_id: string
  is_fraud: boolean
  fraud_score: number
  reasons?: string[]
  score_breakdown?: BackendRiskSignal[]
  card_baseline?: BackendCardBaseline
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

type CsvTransaction = {
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
}

type AnalyzedCsvTransaction = CsvTransaction & {
  fraud_reasons: string[]
  fraud_score: number
  is_fraud: boolean
  review_decision?: ReviewDecision
  reviewed_at?: string
  reviewer_notes?: string
  score_breakdown?: BackendRiskSignal[]
}

type BackendAnalyzedTransaction = CsvTransaction & {
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

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? '/api'

export async function uploadTransactionsCsv(
  file: File,
): Promise<ReviewDataResult> {
  const csvTransactions = parseTransactionsCsv(await file.text())
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

  const analysisResponse = await fetch(
    `${apiBaseUrl}/analysis/all/${uploadPayload.file_hash}`,
  )

  if (!analysisResponse.ok) {
    throw new Error(
      await getErrorMessage(analysisResponse, 'Analysis failed'),
    )
  }

  const analysis = (await analysisResponse.json()) as BackendFraudAnalysis[]

  return {
    fileHash: uploadPayload.file_hash,
    items: mergeTransactionsWithAnalysis(csvTransactions, analysis),
    source: 'upload',
  }
}

export async function fetchReviewDataByHash(
  fileHash: string,
): Promise<ReviewDataResult> {
  const exportResponse = await fetch(`${apiBaseUrl}/export/${fileHash}`)

  if (!exportResponse.ok) {
    throw new Error(await getErrorMessage(exportResponse, 'Cached result failed'))
  }

  const csv = await exportResponse.text()

  return {
    fileHash,
    items: mapAnalyzedTransactions(parseAnalyzedTransactionsCsv(csv)),
    source: 'cache',
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
}: {
  cardId: string
  fileHash: string
}): Promise<CardAnalysis> {
  const response = await fetch(
    `${apiBaseUrl}/analysis/user/${fileHash}/${encodeURIComponent(cardId)}`,
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

function parseTransactionsCsv(csv: string): CsvTransaction[] {
  const rows = parseCsvRows(csv)

  if (rows.length < 2) {
    throw new Error('CSV file is empty')
  }

  const [headers, ...dataRows] = rows
  const headerIndex = new Map(headers.map((header, index) => [header, index]))
  const requiredHeaders = [
    'transaction_id',
    'timestamp',
    'card_id',
    'amount',
    'merchant_name',
    'merchant_category',
    'channel',
    'cardholder_country',
    'merchant_country',
  ]

  const missingHeaders = requiredHeaders.filter(
    (header) => !headerIndex.has(header),
  )

  if (missingHeaders.length > 0) {
    throw new Error(`CSV is missing required columns: ${missingHeaders.join(', ')}`)
  }

  return dataRows
    .filter((row) => row.some((value) => value.trim()))
    .map((row) => {
      const get = (header: string) => row[headerIndex.get(header) ?? -1] ?? ''
      const amount = Number(get('amount'))

      if (!Number.isFinite(amount)) {
        throw new Error(`Invalid amount for transaction ${get('transaction_id')}`)
      }

      return {
        amount,
        card_id: get('card_id'),
        cardholder_country: get('cardholder_country'),
        channel: normalizeChannel(get('channel')),
        device_id: emptyToUndefined(get('device_id')),
        ip_address: emptyToUndefined(get('ip_address')),
        merchant_category: get('merchant_category'),
        merchant_country: get('merchant_country'),
        merchant_name: get('merchant_name'),
        timestamp: get('timestamp'),
        transaction_id: get('transaction_id'),
      }
    })
}

function parseAnalyzedTransactionsCsv(csv: string): AnalyzedCsvTransaction[] {
  const rows = parseCsvRows(csv)

  if (rows.length < 2) {
    throw new Error('Cached result is empty')
  }

  const [headers, ...dataRows] = rows
  const headerIndex = new Map(headers.map((header, index) => [header, index]))
  const requiredHeaders = [
    'transaction_id',
    'timestamp',
    'card_id',
    'amount',
    'merchant_name',
    'merchant_category',
    'channel',
    'cardholder_country',
    'merchant_country',
    'is_fraud',
    'fraud_score',
    'fraud_reasons',
  ]
  const missingHeaders = requiredHeaders.filter(
    (header) => !headerIndex.has(header),
  )

  if (missingHeaders.length > 0) {
    throw new Error(
      `Cached result is missing required columns: ${missingHeaders.join(', ')}`,
    )
  }

  return dataRows
    .filter((row) => row.some((value) => value.trim()))
    .map((row) => {
      const get = (header: string) => row[headerIndex.get(header) ?? -1] ?? ''
      const getOptional = (header: string) =>
        headerIndex.has(header) ? get(header) : ''
      const amount = Number(get('amount'))
      const fraudScore = Number(get('fraud_score'))

      if (!Number.isFinite(amount)) {
        throw new Error(`Invalid amount for transaction ${get('transaction_id')}`)
      }

      if (!Number.isFinite(fraudScore)) {
        throw new Error(
          `Invalid fraud score for transaction ${get('transaction_id')}`,
        )
      }

      return {
        amount,
        card_id: get('card_id'),
        cardholder_country: get('cardholder_country'),
        channel: normalizeChannel(get('channel')),
        device_id: emptyToUndefined(get('device_id')),
        fraud_reasons: parseFraudReasons(get('fraud_reasons')),
        fraud_score: fraudScore,
        ip_address: emptyToUndefined(get('ip_address')),
        is_fraud: parseBoolean(get('is_fraud')),
        merchant_category: get('merchant_category'),
        merchant_country: get('merchant_country'),
        merchant_name: get('merchant_name'),
        review_decision: parseReviewDecision(getOptional('review_decision')),
        reviewed_at: emptyToUndefined(getOptional('reviewed_at')),
        reviewer_notes: emptyToUndefined(getOptional('reviewer_notes')),
        score_breakdown: parseScoreBreakdown(getOptional('score_breakdown')),
        timestamp: get('timestamp'),
        transaction_id: get('transaction_id'),
      }
    })
}

function parseCsvRows(csv: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let inQuotes = false

  for (let index = 0; index < csv.length; index += 1) {
    const char = csv[index]
    const nextChar = csv[index + 1]

    if (char === '"') {
      if (inQuotes && nextChar === '"') {
        field += '"'
        index += 1
      } else {
        inQuotes = !inQuotes
      }
      continue
    }

    if (char === ',' && !inQuotes) {
      row.push(field)
      field = ''
      continue
    }

    if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && nextChar === '\n') {
        index += 1
      }
      row.push(field)
      rows.push(row)
      row = []
      field = ''
      continue
    }

    field += char
  }

  if (field || row.length > 0) {
    row.push(field)
    rows.push(row)
  }

  return rows
}

function mergeTransactionsWithAnalysis(
  transactions: CsvTransaction[],
  analysis: BackendFraudAnalysis[],
): TransactionFlag[] {
  const analysisById = new Map(
    analysis.map((item) => [item.transaction_id, item]),
  )
  const contextByCard = buildCardContext(transactions)

  return transactions
    .map((transaction) => {
      const fraudAnalysis = analysisById.get(transaction.transaction_id)

      if (!fraudAnalysis) {
        return null
      }

      return toTransactionFlag(
        transaction,
        fraudAnalysis,
        contextByCard.get(transaction.card_id),
      )
    })
    .filter(
      (
        transaction,
      ): transaction is TransactionFlag & { isFraudCandidate: boolean } =>
        transaction !== null &&
        (transaction.isFraudCandidate || transaction.score > 0),
    )
    .map(stripFraudCandidateMarker)
    .sort((first, second) => second.score - first.score)
}

function mapAnalyzedTransactions(
  transactions: AnalyzedCsvTransaction[],
): TransactionFlag[] {
  const contextByCard = buildCardContext(transactions)

  return transactions
    .map((transaction) =>
      toTransactionFlag(
        transaction,
        {
          fraud_score: transaction.fraud_score,
          is_fraud: transaction.is_fraud,
          score_breakdown: transaction.score_breakdown,
          reasons: transaction.fraud_reasons,
          transaction_id: transaction.transaction_id,
        },
        contextByCard.get(transaction.card_id),
        transaction.review_decision,
        transaction.reviewed_at,
        transaction.reviewer_notes,
      ),
    )
    .filter(
      (transaction) => transaction.isFraudCandidate || transaction.score > 0,
    )
    .map(stripFraudCandidateMarker)
    .sort((first, second) => second.score - first.score)
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

function stripFraudCandidateMarker({
  isFraudCandidate,
  ...transaction
}: TransactionFlag & { isFraudCandidate: boolean }): TransactionFlag {
  void isFraudCandidate
  return transaction
}

function toTransactionFlag(
  transaction: CsvTransaction,
  analysis: BackendFraudAnalysis,
  cardContext = {
    medianAmount: 0,
    previousTransactions: 0,
    usualCategories: [] as string[],
    usualCountries: [] as string[],
  },
  decision: ReviewDecision = 'pending',
  reviewedAt?: string,
  reviewerNotes?: string,
): TransactionFlag & { isFraudCandidate: boolean } {
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
    deviceId: transaction.device_id,
    ipAddress: transaction.ip_address,
    isFraudCandidate: analysis.is_fraud,
    label: getRiskLabel(analysis.fraud_score),
    merchantCategory: transaction.merchant_category,
    merchantCountry: transaction.merchant_country,
    merchantName: transaction.merchant_name,
    reasons: buildReasons(analysis),
    reviewedAt,
    reviewerNotes,
    isFraud: analysis.is_fraud,
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

function buildCardContext(transactions: CsvTransaction[]) {
  const grouped = new Map<string, CsvTransaction[]>()

  for (const transaction of transactions) {
    grouped.set(transaction.card_id, [
      ...(grouped.get(transaction.card_id) ?? []),
      transaction,
    ])
  }

  return new Map(
    Array.from(grouped.entries()).map(([cardId, cardTransactions]) => {
      const amounts = cardTransactions
        .map((transaction) => transaction.amount)
        .sort((first, second) => first - second)
      const mid = Math.floor(amounts.length / 2)
      const medianAmount =
        amounts.length % 2 === 0
          ? (amounts[mid - 1] + amounts[mid]) / 2
          : amounts[mid]

      return [
        cardId,
        {
          medianAmount,
          previousTransactions: cardTransactions.length,
          usualCategories: getMostCommon(
            cardTransactions.map((transaction) => transaction.merchant_category),
          ),
          usualCountries: getMostCommon(
            cardTransactions.map((transaction) => transaction.merchant_country),
          ),
        },
      ]
    }),
  )
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

function buildReasons(analysis: BackendFraudAnalysis): RiskReason[] {
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

function decisionToBackendAction(
  decision: ReviewDecision,
) {
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

function normalizeChannel(channel: string): CsvTransaction['channel'] {
  if (channel === 'online' || channel === 'in_person' || channel === 'atm') {
    return channel
  }

  throw new Error(`Invalid channel: ${channel}`)
}

function emptyToUndefined(value: string) {
  return value.trim() === '' ? undefined : value
}

function parseBoolean(value: string) {
  return value.trim().toLowerCase() === 'true'
}

function parseFraudReasons(value: string) {
  return value
    .split(';')
    .map((reason) => reason.trim())
    .filter(Boolean)
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

function parseScoreBreakdown(value: string): BackendRiskSignal[] | undefined {
  if (!value.trim()) {
    return undefined
  }

  try {
    const parsedValue = JSON.parse(value) as unknown
    return Array.isArray(parsedValue)
      ? parsedValue.filter(isBackendRiskSignal)
      : undefined
  } catch {
    return undefined
  }
}

function isBackendRiskSignal(value: unknown): value is BackendRiskSignal {
  return isRecord(value)
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
    return parseFraudReasons(value)
  }

  return []
}

function readRiskSignals(record: Record<string, unknown>, key: string) {
  const value = record[key]

  if (Array.isArray(value)) {
    return value.filter(isBackendRiskSignal)
  }

  if (typeof value === 'string') {
    return parseScoreBreakdown(value)
  }

  return undefined
}
