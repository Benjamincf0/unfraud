import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  EmptyTransactionDetail,
  TransactionDetail,
} from "./review/TransactionDetail";
import { QueueList } from "./review/QueueList";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Slider } from "./ui/slider";
import { Tabs } from "./ui/tabs";
import {
  applyScorerDetailToTransaction,
  fetchCardAnalysis,
  fetchRelatedTransactions,
  fetchReviewBootstrap,
  fetchReviewLog,
  fetchReviewQueuePage,
  fetchTransactionDetail,
  submitReviewDecision,
} from "../api/review";
import type { ReviewSession } from "../lib/reviewSessions";
import { mergeTransactionMaps } from "../lib/reviewMemory";
import {
  buildTransactionIndex,
  defaultRiskTuning,
  DETAIL_PREFETCH_CONCURRENCY,
  getEffectiveRiskThreshold,
  QUEUE_PAGE_SIZE,
  resolveRiskSortMode,
  sortTransactionsByScore,
  type ReviewSessionData,
  type RiskSortMode,
  type RiskTuningByMode,
} from "../lib/scoringViews";
import type {
  DecisionAction,
  ReviewDecision,
  SearchFieldKey,
  ReviewLogEntry,
  TransactionFlag,
} from "../types";

type QueueFilter = "pending" | "all" | "approved" | "dismissed" | "escalated";

function isFlaggedTransaction(transaction: TransactionFlag) {
  return transaction.isFraud || transaction.score > 0;
}

function flaggedTransactionsFrom(
  byId: Map<string, TransactionFlag>,
): TransactionFlag[] {
  return Array.from(byId.values()).filter(isFlaggedTransaction);
}

function flaggedTransactionIds(items: TransactionFlag[]) {
  return items
    .filter(isFlaggedTransaction)
    .map((transaction) => transaction.transactionId);
}

function addFlaggedTransactionIds(
  current: Set<string>,
  items: TransactionFlag[],
): Set<string> {
  const next = new Set(current);
  for (const transaction of items) {
    if (isFlaggedTransaction(transaction)) {
      next.add(transaction.transactionId);
    }
  }
  return next;
}

function addFlaggedTransactionId(
  current: Set<string>,
  transaction: TransactionFlag,
): Set<string> {
  if (!isFlaggedTransaction(transaction) || current.has(transaction.transactionId)) {
    return current;
  }

  const next = new Set(current);
  next.add(transaction.transactionId);
  return next;
}

function addQueueTransactionId(
  current: Set<string>,
  transactionId: string,
): Set<string> {
  if (current.has(transactionId)) {
    return current;
  }

  const next = new Set(current);
  next.add(transactionId);
  return next;
}

function queueTransactionIds(items: TransactionFlag[]) {
  return items.map((transaction) => transaction.transactionId);
}

function addQueueTransactionIds(
  current: Set<string>,
  transactionIds: string[],
): Set<string> {
  return transactionIds.reduce(
    (next, transactionId) => addQueueTransactionId(next, transactionId),
    current,
  );
}

type SearchMode = "all" | "single" | "custom";

type ReviewQueueProps = {
  activeFileHash: string;
  onReset: () => void;
  onSelectSession: (fileHash: string) => void;
  session: ReviewSessionData;
  sessions: ReviewSession[];
};

type ReviewSyncVariables = {
  decision: ReviewDecision;
  fileHash: string;
  previousDecision: ReviewDecision;
  rollbackHistory: (history: DecisionAction[]) => DecisionAction[];
  transactionId: string;
};

const filterOptions: Array<{ value: QueueFilter; label: string }> = [
  { value: "pending", label: "Pending" },
  { value: "all", label: "All" },
  { value: "approved", label: "Approved" },
  { value: "dismissed", label: "Dismissed" },
  { value: "escalated", label: "Escalated" },
];

const shortcutOptions = [
  { keys: "J / Down", label: "Next transaction" },
  { keys: "K / Up", label: "Previous transaction" },
  { keys: "A", label: "Approve" },
  { keys: "D", label: "Dismiss" },
  { keys: "E", label: "Escalate" },
  { keys: "U", label: "Undo" },
];

const SEARCH_FIELDS: Array<{
  key: SearchFieldKey;
  label: string;
  values: (transaction: TransactionFlag) => string[];
}> = [
  {
    key: "transaction_id",
    label: "transaction_id",
    values: (transaction) => [transaction.transactionId],
  },
  {
    key: "timestamp",
    label: "timestamp",
    values: (transaction) => {
      const values = [transaction.timestamp];
      const date = new Date(transaction.timestamp);
      if (!Number.isNaN(date.getTime())) {
        values.push(date.toISOString().slice(0, 10));
        values.push(
          date.toLocaleDateString(undefined, {
            day: "2-digit",
            month: "short",
            year: "numeric",
          }),
        );
      }

      return values;
    },
  },
  {
    key: "card_id",
    label: "card_id",
    values: (transaction) => [transaction.cardId],
  },
  {
    key: "amount",
    label: "amount",
    values: (transaction) => [
      String(transaction.amount),
      transaction.amount.toFixed(2),
      `$${transaction.amount.toFixed(2)}`,
    ],
  },
  {
    key: "merchant_name",
    label: "merchant_name",
    values: (transaction) => [transaction.merchantName],
  },
  {
    key: "merchant_category",
    label: "merchant_category",
    values: (transaction) => [transaction.merchantCategory],
  },
  {
    key: "channel",
    label: "channel",
    values: (transaction) => [transaction.channel],
  },
  {
    key: "cardholder_country",
    label: "cardholder_country",
    values: (transaction) => [transaction.cardholderCountry],
  },
  {
    key: "merchant_country",
    label: "merchant_country",
    values: (transaction) => [transaction.merchantCountry],
  },
  {
    key: "device_id",
    label: "device_id",
    values: (transaction) => [transaction.deviceId ?? ""],
  },
  {
    key: "ip_address",
    label: "ip_address",
    values: (transaction) => [transaction.ipAddress ?? ""],
  },
];

const SEARCH_FIELD_MAP = new Map(
  SEARCH_FIELDS.map((field) => [field.key, field]),
);
const sortModeOptions: Array<{ value: RiskSortMode; label: string }> = [
  { value: "active", label: "Active scoring" },
  { value: "heuristic", label: "Heuristic risk" },
  { value: "model", label: "Model risk" },
];

function AuditLog({
  entries,
  error,
  isLoading,
  onClose,
  onSelectTransaction,
  reviewableTransactionIds,
}: {
  entries: ReviewLogEntry[];
  error: string | null;
  isLoading: boolean;
  onClose: () => void;
  onSelectTransaction: (transactionId: string) => void;
  reviewableTransactionIds: Set<string>;
}) {
  return (
    <aside className="audit-log" aria-label="Review audit log">
      <div className="audit-log-header">
        <div>
          <strong>Audit log</strong>
          <span>
            {error
              ? "Could not load"
              : isLoading
                ? "Refreshing"
                : `${entries.length} decisions`}
          </span>
        </div>
        <Button
          aria-label="Close audit log"
          onClick={onClose}
          size="icon"
          title="Close audit log"
          variant="ghost"
        >
          ×
        </Button>
      </div>
      <div className="audit-log-list">
        {entries.length === 0 ? (
          <span className="audit-log-empty">No review decisions yet.</span>
        ) : (
          entries.map((entry) => {
            const canSelect = reviewableTransactionIds.has(entry.transactionId);

            return (
              <button
                className="audit-log-row"
                disabled={!canSelect}
                key={`${entry.transactionId}-${entry.reviewedAt}`}
                onClick={() => onSelectTransaction(entry.transactionId)}
                type="button"
              >
                <strong>{entry.action}</strong>
                <span>{entry.transactionId}</span>
                <time dateTime={entry.reviewedAt}>
                  {formatAuditTime(entry.reviewedAt)}
                </time>
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}

function formatAuditTime(value: string) {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function ReviewQueue({
  activeFileHash,
  onReset,
  onSelectSession,
  session,
  sessions,
}: ReviewQueueProps) {
  const fileHash = session.fileHash;
  const { summary } = session;
  const queryClient = useQueryClient();
  const [useModel, setUseModel] = useState(false);
  const [sortMode, setSortMode] = useState<RiskSortMode>("active");
  const [riskTuningByMode, setRiskTuningByMode] =
    useState<RiskTuningByMode>(defaultRiskTuning);
  const [heuristicById, setHeuristicById] = useState(
    () => new Map<string, TransactionFlag>(),
  );
  const [modelById, setModelById] = useState(
    () => new Map<string, TransactionFlag>(),
  );
  const [relatedById, setRelatedById] = useState(
    () => new Map<string, TransactionFlag[]>(),
  );
  const [loadedHeuristicIds, setLoadedHeuristicIds] = useState(
    () => new Set<string>(),
  );
  const [loadedModelIds, setLoadedModelIds] = useState(() => new Set<string>());
  const [loadedQueueIds, setLoadedQueueIds] = useState(() => new Set<string>());
  const [activeId, setActiveId] = useState("");
  const [filter, setFilter] = useState<QueueFilter>("pending");
  const [query, setQuery] = useState("");
  const [searchMode, setSearchMode] = useState<SearchMode>("all");
  const [singleField, setSingleField] =
    useState<SearchFieldKey>("transaction_id");
  const [customFields, setCustomFields] = useState<SearchFieldKey[]>([
    "transaction_id",
    "card_id",
    "merchant_name",
  ]);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [auditOpen, setAuditOpen] = useState(false);
  const [history, setHistory] = useState<DecisionAction[]>([]);
  const [networkFocus, setNetworkFocus] = useState<{
    label: string;
    transactionIds: Set<string>;
  } | null>(null);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [enrichedTransactionIds, setEnrichedTransactionIds] = useState(
    () => new Set<string>(),
  );
  const [enrichmentFailedIds, setEnrichmentFailedIds] = useState(
    () => new Set<string>(),
  );
  const [enrichingTransactionId, setEnrichingTransactionId] = useState<
    string | null
  >(null);
  const heuristicByIdRef = useRef(heuristicById);
  const modelByIdRef = useRef(modelById);
  const relatedByIdRef = useRef(relatedById);
  const enrichedTransactionIdsRef = useRef(enrichedTransactionIds);
  const detailInFlightRef = useRef(new Map<string, Promise<boolean>>());
  heuristicByIdRef.current = heuristicById;
  modelByIdRef.current = modelById;
  relatedByIdRef.current = relatedById;
  enrichedTransactionIdsRef.current = enrichedTransactionIds;
  const bootstrapQuery = useQuery({
    queryFn: () => fetchReviewBootstrap(fileHash, summary),
    queryKey: ["review-bootstrap", fileHash],
  });
  const {
    error: reviewSyncError,
    isError: reviewSyncFailed,
    mutate: syncReviewDecision,
  } = useMutation<unknown, Error, ReviewSyncVariables>({
    mutationFn: ({
      previousDecision: _previousDecision,
      rollbackHistory: _rollbackHistory,
      ...variables
    }) => submitReviewDecision(variables),
    onError: (_error, variables) => {
      updateDecision(variables.transactionId, variables.previousDecision);
      setHistory(variables.rollbackHistory);
      setActiveId(variables.transactionId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["review-log", fileHash] });
    },
  });
  const reviewLogQuery = useQuery({
    enabled: Boolean(fileHash),
    queryFn: () => fetchReviewLog(fileHash),
    queryKey: ["review-log", fileHash],
  });

  const updateDecision = useCallback(
    (transactionId: string, decision: ReviewDecision) => {
      const applyDecision = (current: Map<string, TransactionFlag>) => {
        const transaction = current.get(transactionId);
        if (!transaction) {
          return current;
        }

        const next = new Map(current);
        next.set(transactionId, { ...transaction, decision });
        return next;
      };

      setHeuristicById(applyDecision);
      setModelById(applyDecision);
    },
    [],
  );

  useEffect(() => {
    setHeuristicById(new Map());
    setModelById(new Map());
    setRelatedById(new Map());
    setLoadedHeuristicIds(new Set());
    setLoadedModelIds(new Set());
    setLoadedQueueIds(new Set());
    setEnrichedTransactionIds(new Set());
    setEnrichmentFailedIds(new Set());
    setEnrichingTransactionId(null);
    detailInFlightRef.current = new Map();
    setUseModel(false);
    setSortMode("active");
    setRiskTuningByMode(defaultRiskTuning);
    setActiveId("");
    setFilter("pending");
    setSearchMode("all");
    setSingleField("transaction_id");
    setCustomFields(["transaction_id", "card_id", "merchant_name"]);
    setNetworkFocus(null);
    setHistory([]);
  }, [fileHash]);

  useEffect(() => {
    if (!bootstrapQuery.data) {
      return;
    }

    setHeuristicById((current) =>
      mergeTransactionMaps(current, bootstrapQuery.data.heuristic),
    );
    setModelById((current) =>
      mergeTransactionMaps(current, bootstrapQuery.data.model),
    );
    const bootstrapHeuristicIds = flaggedTransactionIds(
      bootstrapQuery.data.heuristic,
    );
    const bootstrapModelIds = flaggedTransactionIds(bootstrapQuery.data.model);
    setLoadedHeuristicIds(new Set(bootstrapHeuristicIds));
    setLoadedModelIds(new Set(bootstrapModelIds));
    setLoadedQueueIds(
      new Set([...bootstrapHeuristicIds, ...bootstrapModelIds]),
    );
    setActiveId((current) => {
      if (current) {
        return current;
      }

      return (
        bootstrapQuery.data.heuristic[0]?.transactionId ||
        bootstrapQuery.data.model[0]?.transactionId ||
        ""
      );
    });
  }, [bootstrapQuery.data]);

  const enrichTransactionDetail = useCallback(
    async (transactionId: string) => {
      if (enrichedTransactionIdsRef.current.has(transactionId)) {
        return true;
      }

      const inFlight = detailInFlightRef.current.get(transactionId);
      if (inFlight) {
        return inFlight;
      }

      const enrichment = (async () => {
        let heuristicItem = heuristicByIdRef.current.get(transactionId);
        if (!heuristicItem) {
          const heuristicPage = await fetchReviewQueuePage(fileHash, {
            flaggedOnly: false,
            transactionId,
            useModel: false,
          });
          heuristicItem = heuristicPage.items[0];
          if (heuristicItem) {
            setHeuristicById((current) =>
              mergeTransactionMaps(current, [heuristicItem!]),
            );
            setLoadedHeuristicIds((current) =>
              addFlaggedTransactionId(current, heuristicItem!),
            );
          }
        }

        let modelItem = modelByIdRef.current.get(transactionId);
        if (summary.mlModelAvailable && !modelItem) {
          const modelPage = await fetchReviewQueuePage(fileHash, {
            flaggedOnly: false,
            transactionId,
            useModel: true,
          });
          modelItem = modelPage.items[0];
          if (modelItem) {
            setModelById((current) =>
              mergeTransactionMaps(current, [modelItem!]),
            );
            setLoadedModelIds((current) =>
              addFlaggedTransactionId(current, modelItem!),
            );
          }
        }

        const detail = await fetchTransactionDetail(fileHash, transactionId);

        setHeuristicById((current) => {
          const existing = current.get(transactionId) ?? heuristicItem;
          if (!existing) {
            return current;
          }

          const next = new Map(current);
          next.set(
            transactionId,
            applyScorerDetailToTransaction(
              existing,
              detail.heuristic,
              transactionId,
            ),
          );
          return next;
        });

        if (detail.model) {
          setModelById((current) => {
            const existing =
              current.get(transactionId) ?? modelItem ?? heuristicItem;
            if (!existing) {
              return current;
            }

            const next = new Map(current);
            next.set(
              transactionId,
              applyScorerDetailToTransaction(
                existing,
                detail.model!,
                transactionId,
              ),
            );
            return next;
          });
        }

        setLoadedQueueIds((current) =>
          addQueueTransactionId(current, transactionId),
        );

        enrichedTransactionIdsRef.current.add(transactionId);
        setEnrichedTransactionIds(
          new Set(enrichedTransactionIdsRef.current),
        );
        setEnrichmentFailedIds((current) => {
          if (!current.has(transactionId)) {
            return current;
          }

          const next = new Set(current);
          next.delete(transactionId);
          return next;
        });
        return true;
      })();

      detailInFlightRef.current.set(transactionId, enrichment);

      try {
        return await enrichment;
      } catch {
        setEnrichmentFailedIds((current) => {
          const next = new Set(current);
          next.add(transactionId);
          return next;
        });
        return false;
      } finally {
        detailInFlightRef.current.delete(transactionId);
      }
    },
    [fileHash, summary.mlModelAvailable],
  );

  const loadRelatedForTransaction = useCallback(
    async (transactionId: string) => {
      if (relatedByIdRef.current.has(transactionId)) {
        return;
      }

      const relatedHeuristic = await fetchRelatedTransactions(
        fileHash,
        transactionId,
        false,
      );

      setRelatedById((current) => {
        const next = new Map(current);
        next.set(transactionId, relatedHeuristic);
        return next;
      });
    },
    [fileHash],
  );

  useEffect(() => {
    if (!bootstrapQuery.data) {
      return;
    }

    let cancelled = false;
    const bootstrapIds = bootstrapQuery.data.heuristic.map(
      (transaction) => transaction.transactionId,
    );

    async function prefetchBootstrapDetails() {
      const [priorityId, ...remainingIds] = bootstrapIds;

      if (priorityId && !cancelled) {
        await enrichTransactionDetail(priorityId).catch(() => false);
      }

      let nextIndex = 0;

      async function worker() {
        while (nextIndex < remainingIds.length && !cancelled) {
          const transactionId = remainingIds[nextIndex];
          nextIndex += 1;

          if (enrichedTransactionIdsRef.current.has(transactionId)) {
            continue;
          }

          await enrichTransactionDetail(transactionId).catch(() => false);
        }
      }

      await Promise.all(
        Array.from({ length: DETAIL_PREFETCH_CONCURRENCY }, () => worker()),
      );
    }

    void prefetchBootstrapDetails();

    return () => {
      cancelled = true;
    };
  }, [bootstrapQuery.data, enrichTransactionDetail]);

  useLayoutEffect(() => {
    if (!activeId) {
      setEnrichingTransactionId(null);
      return;
    }

    if (enrichedTransactionIds.has(activeId)) {
      setEnrichingTransactionId(null);
      return;
    }

    setEnrichingTransactionId(activeId);
  }, [activeId, enrichedTransactionIds]);

  useEffect(() => {
    if (!activeId) {
      return;
    }

    let cancelled = false;

    async function onActiveTransactionChange() {
      const needsDetail = !enrichedTransactionIdsRef.current.has(activeId);

      if (needsDetail) {
        await enrichTransactionDetail(activeId);
        if (!cancelled) {
          setEnrichingTransactionId(null);
        }
      }

      if (!cancelled) {
        await loadRelatedForTransaction(activeId).catch(() => undefined);
      }
    }

    void onActiveTransactionChange();

    return () => {
      cancelled = true;
    };
  }, [activeId, enrichTransactionDetail, loadRelatedForTransaction]);

  const activeFlaggedCount = useModel
    ? summary.modelFlaggedCount
    : summary.flaggedCount;

  const transactions = useMemo(
    () =>
      flaggedTransactionsFrom(useModel ? modelById : heuristicById),
    [heuristicById, modelById, useModel],
  );

  const loadedHeuristicCount = loadedHeuristicIds.size;
  const loadedModelCount = loadedModelIds.size;
  const loadedFlaggedCount = loadedQueueIds.size;

  const heuristicIndex = useMemo(
    () => buildTransactionIndex(Array.from(heuristicById.values())),
    [heuristicById],
  );
  const modelIndex = useMemo(
    () => buildTransactionIndex(Array.from(modelById.values())),
    [modelById],
  );
  const resolvedSortMode = useMemo(
    () => resolveRiskSortMode(sortMode, useModel),
    [sortMode, useModel],
  );
  const scoreIndex = useMemo(() => {
    const source =
      resolvedSortMode === "model" ? modelIndex : heuristicIndex;

    return new Map(
      Array.from(source.entries()).map(([transactionId, transaction]) => [
        transactionId,
        transaction.score,
      ]),
    );
  }, [heuristicIndex, modelIndex, resolvedSortMode]);
  const activeTuning = riskTuningByMode[sortMode];
  const effectiveRiskThreshold = useMemo(
    () =>
      getEffectiveRiskThreshold(
        activeTuning.riskThreshold,
        activeTuning.falsePositiveCost,
      ),
    [activeTuning.falsePositiveCost, activeTuning.riskThreshold],
  );
  const orderedTransactions = useMemo(
    () => sortTransactionsByScore(transactions, scoreIndex),
    [scoreIndex, transactions],
  );
  const networkTransactions = useMemo(() => {
    if (!activeId) {
      return Array.from(heuristicById.values());
    }

    return relatedById.get(activeId) ?? Array.from(heuristicById.values());
  }, [activeId, heuristicById, relatedById]);

  const searchScopeKeys = useMemo(() => {
    if (searchMode === "single") {
      return [singleField];
    }

    if (searchMode === "custom") {
      return customFields.length > 0
        ? customFields
        : SEARCH_FIELDS.map((field) => field.key);
    }

    return SEARCH_FIELDS.map((field) => field.key);
  }, [customFields, searchMode, singleField]);

  const visibleEntries = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();

    return orderedTransactions
      .map((transaction) => {
        const filterPasses =
          filter === "all" ? true : transaction.decision === filter;
        const thresholdScore =
          scoreIndex.get(transaction.transactionId) ?? transaction.score;
        const thresholdPasses = thresholdScore * 100 >= effectiveRiskThreshold;
        const networkPasses = networkFocus
          ? networkFocus.transactionIds.has(transaction.transactionId)
          : true;

        let matchedFieldLabels: string[] = [];

        if (normalizedQuery) {
          matchedFieldLabels = searchScopeKeys
            .flatMap((key) => {
              const field = SEARCH_FIELD_MAP.get(key);
              if (!field) {
                return [];
              }

              const hasMatch = field
                .values(transaction)
                .some((value) => value.toLowerCase().includes(normalizedQuery));

              return hasMatch ? [field.label] : [];
            })
            .filter((value, index, array) => array.indexOf(value) === index);
        }

        const queryPasses = !normalizedQuery || matchedFieldLabels.length > 0;

        return {
          matchedFieldLabels,
          transaction,
          visible:
            filterPasses && thresholdPasses && networkPasses && queryPasses,
        };
      })
      .filter((entry) => entry.visible);
  }, [
    effectiveRiskThreshold,
    filter,
    networkFocus,
    orderedTransactions,
    query,
    scoreIndex,
    searchScopeKeys,
  ]);

  const visibleTransactions = useMemo(
    () => visibleEntries.map((entry) => entry.transaction),
    [visibleEntries],
  );

  const matchFieldsByTransactionId = useMemo(
    () =>
      new Map(
        visibleEntries.map((entry) => [
          entry.transaction.transactionId,
          entry.matchedFieldLabels,
        ]),
      ),
    [visibleEntries],
  );

  const activeTransaction =
    visibleTransactions.find(
      (transaction) => transaction.transactionId === activeId,
    ) ?? visibleTransactions[0];
  const isReasonsLoading = Boolean(
    activeTransaction &&
      enrichingTransactionId === activeTransaction.transactionId,
  );
  const reasonsLoadError = activeTransaction
    ? enrichmentFailedIds.has(activeTransaction.transactionId)
      ? "Could not load the risk breakdown for this transaction."
      : null
    : null;

  const reviewableTransactionIds = useMemo(
    () => new Set(transactions.map((transaction) => transaction.transactionId)),
    [transactions],
  );

  const activeCardAnalysisQuery = useQuery({
    enabled: Boolean(activeTransaction?.cardId),
    queryFn: () =>
      fetchCardAnalysis({
        cardId: activeTransaction?.cardId ?? "",
        fileHash,
        useModel,
      }),
    queryKey: ["card-analysis", fileHash, activeTransaction?.cardId, useModel],
  });

  const queueStats = useMemo(() => {
    return transactions.reduce(
      (stats, transaction) => {
        stats[transaction.decision] += 1;
        return stats;
      },
      { approved: 0, dismissed: 0, escalated: 0, pending: 0 } as Record<
        ReviewDecision,
        number
      >,
    );
  }, [transactions]);

  const queueFilterOptions = useMemo(
    () =>
      filterOptions.map((option) => {
        const count =
          option.value === "all" ? transactions.length : queueStats[option.value];

        return {
          ...option,
          label: `${option.label} (${count})`,
        };
      }),
    [queueStats, transactions.length],
  );

  const totalScoredCount = summary.totalTransactions;
  const queuedCount = activeFlaggedCount;
  const notQueuedCount = Math.max(0, totalScoredCount - queuedCount);
  const reviewedCount = transactions.length - queueStats.pending;
  const scorerLabel = useModel ? "ML model" : "Heuristic";
  const loadedScorerCount = useModel ? loadedModelCount : loadedHeuristicCount;
  const hasMoreFlagged = loadedScorerCount < activeFlaggedCount;

  const statusContextLine = useMemo(() => {
    const filtersActive =
      filter !== "pending" ||
      query.trim() !== "" ||
      networkFocus !== null ||
      effectiveRiskThreshold > 0;

    if (loadedFlaggedCount < activeFlaggedCount) {
      return `Loaded ${loadedFlaggedCount.toLocaleString()} of ${activeFlaggedCount.toLocaleString()} flagged (${scorerLabel.toLowerCase()})`;
    }

    if (filtersActive && visibleTransactions.length !== transactions.length) {
      return `Showing ${visibleTransactions.length.toLocaleString()} of ${transactions.length.toLocaleString()} loaded flagged (filters active)`;
    }

    if (reviewedCount > 0) {
      return `${queueStats.pending.toLocaleString()} still to review · ${(transactions.length - queueStats.pending).toLocaleString()} reviewed`;
    }

    if (notQueuedCount > 0) {
      return `${notQueuedCount.toLocaleString()} scored with no flag (not in this list)`;
    }

    return null;
  }, [
    activeFlaggedCount,
    effectiveRiskThreshold,
    filter,
    loadedFlaggedCount,
    networkFocus,
    notQueuedCount,
    queueStats.pending,
    query,
    reviewedCount,
    scorerLabel,
    transactions.length,
    visibleTransactions.length,
  ]);

  const handleUseModelChange = (nextUseModel: boolean) => {
    if (nextUseModel && !summary.mlModelAvailable) {
      return;
    }

    setUseModel(nextUseModel);
  };

  const loadMoreFlagged = useCallback(async () => {
    const canLoadHeuristic = loadedHeuristicCount < summary.flaggedCount;
    const canLoadModel =
      summary.mlModelAvailable &&
      loadedModelCount < summary.modelFlaggedCount;

    if (isLoadingMore || (!canLoadHeuristic && !canLoadModel)) {
      return;
    }

    setIsLoadingMore(true);

    try {
      let prefetchIds: string[] = [];

      if (canLoadHeuristic) {
        const heuristicPage = await fetchReviewQueuePage(fileHash, {
          limit: QUEUE_PAGE_SIZE,
          offset: loadedHeuristicCount,
          useModel: false,
        });
        setHeuristicById((current) =>
          mergeTransactionMaps(current, heuristicPage.items),
        );
        setLoadedHeuristicIds((current) =>
          addFlaggedTransactionIds(current, heuristicPage.items),
        );
        setLoadedQueueIds((current) =>
          addQueueTransactionIds(current, queueTransactionIds(heuristicPage.items)),
        );
        prefetchIds = heuristicPage.items.map(
          (transaction) => transaction.transactionId,
        );
      }

      if (canLoadModel) {
        const modelPage = await fetchReviewQueuePage(fileHash, {
          limit: QUEUE_PAGE_SIZE,
          offset: loadedModelCount,
          useModel: true,
        });
        setModelById((current) =>
          mergeTransactionMaps(current, modelPage.items),
        );
        setLoadedModelIds((current) =>
          addFlaggedTransactionIds(current, modelPage.items),
        );
        setLoadedQueueIds((current) =>
          addQueueTransactionIds(current, queueTransactionIds(modelPage.items)),
        );

        if (prefetchIds.length === 0) {
          prefetchIds = modelPage.items.map(
            (transaction) => transaction.transactionId,
          );
        }
      }

      void (async () => {
        let nextIndex = 0;
        const transactionIds = prefetchIds;

        async function worker() {
          while (nextIndex < transactionIds.length) {
            const transactionId = transactionIds[nextIndex];
            nextIndex += 1;

            if (enrichedTransactionIdsRef.current.has(transactionId)) {
              continue;
            }

            await enrichTransactionDetail(transactionId).catch(() => false);
          }
        }

        await Promise.all(
          Array.from({ length: DETAIL_PREFETCH_CONCURRENCY }, () => worker()),
        );
      })();
    } finally {
      setIsLoadingMore(false);
    }
  }, [
    enrichTransactionDetail,
    fileHash,
    isLoadingMore,
    loadedHeuristicIds,
    loadedModelIds,
    loadedQueueIds,
    summary.flaggedCount,
    summary.mlModelAvailable,
    summary.modelFlaggedCount,
  ]);

  useEffect(() => {
    if (sortMode === "model" && !summary.mlModelAvailable) {
      setSortMode("active");
    }
  }, [sortMode, summary.mlModelAvailable]);

  const updateRiskTuning = (
    mode: RiskSortMode,
    patch: Partial<RiskTuningByMode[RiskSortMode]>,
  ) => {
    setRiskTuningByMode((current) => ({
      ...current,
      [mode]: {
        ...current[mode],
        ...patch,
      },
    }));
  };

  const focusRelatedTransactions = useCallback(
    ({
      label,
      transactionIds,
    }: {
      label: string;
      transactionIds: string[];
    }) => {
      if (transactionIds.length === 0) {
        return;
      }

      const ids = new Set(transactionIds);
      const firstVisible = transactions.find((tx) => ids.has(tx.transactionId));

      setFilter("all");
      setQuery("");
      setNetworkFocus({ label, transactionIds: ids });
      if (firstVisible) {
        setActiveId(firstVisible.transactionId);
      }
    },
    [transactions],
  );

  const filterByTransactionField = useCallback(
    ({ field, value }: { field: SearchFieldKey; value: string }) => {
      const queryValue = value.trim();

      if (!queryValue) {
        return;
      }

      const firstMatch = transactions.find((transaction) => {
        const searchField = SEARCH_FIELD_MAP.get(field);
        return searchField
          ?.values(transaction)
          .some((candidate) =>
            candidate.toLowerCase().includes(queryValue.toLowerCase()),
          );
      });

      setFilter("all");
      setNetworkFocus(null);
      setSearchMode("single");
      setSingleField(field);
      setQuery(queryValue);

      if (firstMatch) {
        setActiveId(firstMatch.transactionId);
      }
    },
    [transactions],
  );

  const filterByCardCountry = useCallback(
    ({ cardId, country }: { cardId: string; country: string }) => {
      const normalizedCountry = country.trim().toUpperCase();

      if (!cardId || !normalizedCountry) {
        return;
      }

      const matchingTransactions = transactions.filter(
        (transaction) =>
          transaction.cardId === cardId &&
          transaction.merchantCountry.trim().toUpperCase() === normalizedCountry,
      );
      const transactionIds = new Set(
        matchingTransactions.map((transaction) => transaction.transactionId),
      );

      setFilter("all");
      setQuery("");
      setNetworkFocus({
        label: `${cardId} in ${normalizedCountry}`,
        transactionIds,
      });
      setSearchMode("all");

      if (matchingTransactions[0]) {
        setActiveId(matchingTransactions[0].transactionId);
      }
    },
    [transactions],
  );

  const toggleCustomField = (key: SearchFieldKey) => {
    setCustomFields((current) =>
      current.includes(key)
        ? current.filter((value) => value !== key)
        : [...current, key],
    );
  };

  const decide = useCallback(
    (
      transactionId: string,
      nextDecision: Exclude<ReviewDecision, "pending">,
    ) => {
      const transaction = transactions.find(
        (item) => item.transactionId === transactionId,
      );

      if (!transaction) {
        return;
      }

      const activeIndex = visibleTransactions.findIndex(
        (item) => item.transactionId === transactionId,
      );
      const nextActiveTransaction =
        visibleTransactions[activeIndex + 1] ??
        visibleTransactions[activeIndex - 1] ??
        null;
      const action: DecisionAction = {
        actedAt: new Date().toISOString(),
        nextDecision,
        previousDecision: transaction.decision,
        transactionId,
      };

      updateDecision(transactionId, nextDecision);
      setHistory((previous) => [action, ...previous]);
      setActiveId(nextActiveTransaction?.transactionId ?? "");
      syncReviewDecision({
        decision: nextDecision,
        fileHash,
        previousDecision: transaction.decision,
        rollbackHistory: (current) =>
          current.filter(
            (historyAction) => historyAction.actedAt !== action.actedAt,
          ),
        transactionId,
      });
    },
    [fileHash, syncReviewDecision, transactions, updateDecision, visibleTransactions],
  );

  const undo = useCallback(() => {
    const [lastAction, ...rest] = history;

    if (!lastAction) {
      return;
    }

    updateDecision(lastAction.transactionId, lastAction.previousDecision);
    setHistory(rest);
    setActiveId(lastAction.transactionId);
    syncReviewDecision({
      decision: lastAction.previousDecision,
      fileHash,
      previousDecision: lastAction.nextDecision,
      rollbackHistory: (current) => [lastAction, ...current],
      transactionId: lastAction.transactionId,
    });
  }, [fileHash, history, syncReviewDecision, updateDecision]);

  const moveActive = useCallback(
    (direction: 1 | -1) => {
      if (!activeTransaction) {
        return;
      }

      const activeIndex = visibleTransactions.findIndex(
        (transaction) =>
          transaction.transactionId === activeTransaction.transactionId,
      );
      const nextIndex = Math.min(
        Math.max(activeIndex + direction, 0),
        visibleTransactions.length - 1,
      );
      setActiveId(visibleTransactions[nextIndex]?.transactionId ?? "");
    },
    [activeTransaction, visibleTransactions],
  );

  useEffect(() => {
    if (activeTransaction) {
      setActiveId(activeTransaction.transactionId);
    }
  }, [activeTransaction]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;

      if (target?.matches("input, textarea, select, button")) {
        return;
      }

      if (event.key === "j" || event.key === "ArrowDown") {
        event.preventDefault();
        moveActive(1);
      }

      if (event.key === "k" || event.key === "ArrowUp") {
        event.preventDefault();
        moveActive(-1);
      }

      if (!activeTransaction) {
        return;
      }

      if (event.key.toLowerCase() === "a") {
        decide(activeTransaction.transactionId, "approved");
      }

      if (event.key.toLowerCase() === "d") {
        decide(activeTransaction.transactionId, "dismissed");
      }

      if (event.key.toLowerCase() === "e") {
        decide(activeTransaction.transactionId, "escalated");
      }

      if (event.key.toLowerCase() === "u") {
        undo();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeTransaction, decide, moveActive, undo]);

  if (bootstrapQuery.isLoading && heuristicById.size === 0) {
    return (
      <div className="app-shell review-shell">
        <main className="workspace">
          <p className="empty-copy">Loading review queue…</p>
        </main>
      </div>
    );
  }

  if (bootstrapQuery.isError && heuristicById.size === 0) {
    return (
      <div className="app-shell review-shell">
        <main className="workspace">
          <p className="empty-copy" role="alert">
            {bootstrapQuery.error instanceof Error
              ? bootstrapQuery.error.message
              : "Could not load review queue."}
          </p>
          <Button onClick={onReset} size="sm" variant="outline">
            Upload CSV
          </Button>
        </main>
      </div>
    );
  }

  return (
    <div className="app-shell review-shell">
      <main className="workspace">
        <header className="topbar">
          <div
            className="review-status"
            aria-label="Review queue summary"
            title="Flagged = scorer marked suspicious or gave a risk score above zero. The upload is fully scored; only flagged rows appear in the list below."
          >
            <p className="review-status-primary">
              <strong>
                {queuedCount.toLocaleString()} /{' '}
                {totalScoredCount.toLocaleString()}
              </strong>
              <span>flagged for review</span>
              <span className="review-status-scorer">{scorerLabel}</span>
            </p>
            {statusContextLine ? (
              <p className="review-status-context">{statusContextLine}</p>
            ) : null}
            {networkFocus || reviewSyncFailed ? (
              <p className="review-status-meta">
                {networkFocus ? (
                  <span>Network: {networkFocus.label}</span>
                ) : null}
                {reviewSyncFailed ? (
                  <span title={reviewSyncError?.message}>Sync failed</span>
                ) : null}
              </p>
            ) : null}
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
                    {session.label} - {session.fileHash.slice(0, 8)}
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
              <label
                className="search-control compact"
                htmlFor="search-mode-select"
              >
                <span className="visually-hidden">Scope</span>
                <select
                  className="search-select"
                  id="search-mode-select"
                  onChange={(event) =>
                    setSearchMode(event.target.value as SearchMode)
                  }
                  value={searchMode}
                >
                  <option value="all">All columns</option>
                  <option value="single">One column</option>
                  <option value="custom">Custom set</option>
                </select>
              </label>

              {searchMode === "single" ? (
                <label
                  className="search-control compact"
                  htmlFor="single-column-select"
                >
                  <span className="visually-hidden">Column</span>
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
                aria-label="Show keyboard shortcuts"
                aria-expanded={shortcutsOpen}
                aria-haspopup="true"
                onClick={() => setShortcutsOpen((open) => !open)}
                size="icon"
                title="Shortcuts"
                variant="outline"
              >
                ?
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
              aria-label="Undo last decision"
              disabled={history.length === 0}
              onClick={undo}
              size="icon"
              title="Undo"
              variant="outline"
            >
              ↶
            </Button>
            <Button
              aria-expanded={auditOpen}
              aria-label="Toggle audit log"
              onClick={() => setAuditOpen((open) => !open)}
              size="sm"
              variant="outline"
            >
              Audit Log
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
            {hasMoreFlagged ? (
              <Button
                disabled={isLoadingMore || bootstrapQuery.isFetching}
                onClick={() => void loadMoreFlagged()}
                size="sm"
                variant="outline"
              >
                {isLoadingMore ? "Loading…" : "Load more flagged"}
              </Button>
            ) : null}
          </div>
        </header>

        <div className="review-controls">
          <Tabs
            className="queue-tabs"
            onValueChange={setFilter}
            options={queueFilterOptions}
            value={filter}
          />

          <div className="tuning-controls" aria-label="Queue tuning controls">
            <div className="tuning-controls-sliders">
              <label className="tuning-control">
                <span>
                  Risk threshold (
                  {sortModeOptions.find((option) => option.value === sortMode)?.label}
                  )
                </span>
                <Slider
                  aria-label="Risk threshold"
                  max={95}
                  min={0}
                  onChange={(event) =>
                    updateRiskTuning(sortMode, {
                      riskThreshold: Number(event.target.value),
                    })
                  }
                  step={5}
                  value={activeTuning.riskThreshold}
                />
                <strong>{Math.round(effectiveRiskThreshold)}%</strong>
              </label>
              <label className="tuning-control">
                <span>False positive cost</span>
                <Slider
                  aria-label="False positive cost"
                  max={9}
                  min={1}
                  onChange={(event) =>
                    updateRiskTuning(sortMode, {
                      falsePositiveCost: Number(event.target.value),
                    })
                  }
                  step={1}
                  value={activeTuning.falsePositiveCost}
                />
                <strong>{activeTuning.falsePositiveCost}</strong>
              </label>
            </div>
            <div
              aria-label="Scoring engine"
              className="scoring-toggle"
              role="group"
            >
              <span
                className={
                  useModel
                    ? "scoring-toggle-label"
                    : "scoring-toggle-label scoring-toggle-label-active"
                }
              >
                Heuristic
              </span>
              <label className="scoring-switch">
                <input
                  checked={useModel}
                  className="scoring-switch-input"
                  disabled={!summary.mlModelAvailable}
                  onChange={(event) =>
                    handleUseModelChange(event.target.checked)
                  }
                  type="checkbox"
                />
                <span className="scoring-switch-track">
                  <span className="scoring-switch-thumb" />
                </span>
              </label>
              <span
                className={
                  useModel
                    ? "scoring-toggle-label scoring-toggle-label-active"
                    : "scoring-toggle-label"
                }
              >
                ML model
              </span>
            </div>
          </div>

          {searchMode === "custom" ? (
            <div
              className="search-custom-fields"
              aria-label="Custom search columns"
            >
              {SEARCH_FIELDS.map((field) => {
                const selected = customFields.includes(field.key);

                return (
                  <button
                    className={
                      selected
                        ? "search-chip search-chip-active"
                        : "search-chip"
                    }
                    key={field.key}
                    onClick={() => toggleCustomField(field.key)}
                    type="button"
                  >
                    {field.label}
                  </button>
                );
              })}
            </div>
          ) : null}
        </div>

        <div
          className={
            auditOpen
              ? "review-layout review-layout-with-audit"
              : "review-layout"
          }
        >
          <QueueList
            activeTransactionId={activeTransaction?.transactionId}
            matchFieldsByTransactionId={matchFieldsByTransactionId}
            onSelect={setActiveId}
            onSortModeChange={setSortMode}
            searchQuery={query}
            sortMode={sortMode}
            sortModeOptions={sortModeOptions}
            sortModeDisabled={!summary.mlModelAvailable}
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
              isReasonsLoading={isReasonsLoading}
              onDecide={decide}
              onFilterCardCountry={filterByCardCountry}
              onFilterByField={filterByTransactionField}
              onFocusRelatedTransactions={focusRelatedTransactions}
              onSelectTransaction={setActiveId}
              reasonsLoadError={reasonsLoadError}
              reviewableTransactionIds={reviewableTransactionIds}
              transactions={networkTransactions}
              transaction={activeTransaction}
            />
          ) : (
            <EmptyTransactionDetail />
          )}

          {auditOpen ? (
            <AuditLog
              entries={reviewLogQuery.data ?? []}
              error={
                reviewLogQuery.error instanceof Error
                  ? reviewLogQuery.error.message
                  : null
              }
              isLoading={reviewLogQuery.isFetching}
              onClose={() => setAuditOpen(false)}
              onSelectTransaction={setActiveId}
              reviewableTransactionIds={reviewableTransactionIds}
            />
          ) : null}
        </div>
      </main>
    </div>
  );
}
