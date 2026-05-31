import { useMemo } from 'react'
import type { TransactionFlag } from '../../types'

type CrossCardNetworkPanelProps = {
  activeTransaction: TransactionFlag
  onFocusRelatedTransactions: (payload: {
    label: string
    transactionIds: string[]
  }) => void
  transactions: TransactionFlag[]
}

type NetworkNode = {
  id: string
  key: string
  label: string
  type: 'card' | 'device' | 'ip'
  x: number
  y: number
  linkedCount: number
  isActive: boolean
}

type NetworkEdge = {
  id: string
  from: string
  to: string
  isActive: boolean
  weight: number
}

const WIDTH = 760
const HEIGHT = 280

export function CrossCardNetworkPanel({
  activeTransaction,
  onFocusRelatedTransactions,
  transactions,
}: CrossCardNetworkPanelProps) {
  const model = useMemo(
    () => buildNetworkModel(activeTransaction, transactions),
    [activeTransaction, transactions],
  )

  if (model.nodes.length === 0) {
    return null
  }

  return (
    <section className="network-panel" aria-label="Cross-card network graph">
      <div className="chart-title-row">
        <strong>Cross-card device/IP network</strong>
      </div>

      <div className="network-legend" aria-hidden="true">
        <span>
          <i className="legend-swatch network-card-swatch" />
          Card
        </span>
        <span>
          <i className="legend-swatch network-device-swatch" />
          Device
        </span>
        <span>
          <i className="legend-swatch network-ip-swatch" />
          IP
        </span>
        <span>
          <i className="legend-swatch network-active-swatch" />
          Active transaction path
        </span>
      </div>

      <div className="network-frame">
        <svg className="network-svg" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img">
          <title>Card-device-IP relationship graph for review context</title>
          {model.edges.map((edge) => {
            const fromNode = model.nodeById.get(edge.from)
            const toNode = model.nodeById.get(edge.to)
            if (!fromNode || !toNode) {
              return null
            }

            const controlX = (fromNode.x + toNode.x) / 2
            const curve = `M ${fromNode.x} ${fromNode.y} Q ${controlX} ${fromNode.y}, ${toNode.x} ${toNode.y}`

            return (
              <path
                key={edge.id}
                className={edge.isActive ? 'network-edge network-edge-active' : 'network-edge'}
                d={curve}
                style={{ strokeWidth: Math.min(4, 1 + edge.weight * 0.6) }}
              />
            )
          })}

          {model.nodes.map((node) => (
            <g
              className="network-node-group"
              key={node.id}
              onClick={() =>
                onFocusRelatedTransactions({
                  label: `${node.type}: ${node.key}`,
                  transactionIds: model.nodeTransactionIds.get(node.id) ?? [],
                })
              }
              onKeyDown={(event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault()
                  onFocusRelatedTransactions({
                    label: `${node.type}: ${node.key}`,
                    transactionIds: model.nodeTransactionIds.get(node.id) ?? [],
                  })
                }
              }}
              role="button"
              tabIndex={0}
              transform={`translate(${node.x}, ${node.y})`}
            >
              <circle
                className={[
                  'network-node',
                  `network-node-${node.type}`,
                  node.isActive ? 'network-node-active' : '',
                ]
                  .filter(Boolean)
                  .join(' ')}
                r={Math.min(18, 10 + node.linkedCount)}
              />
              <text className="network-label" x={0} y={28}>
                {node.label}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <div className="network-summary">
        <span>
          Related cards: <strong>{model.relatedCardCount}</strong>
        </span>
        <span>
          Shared devices: <strong>{model.sharedDeviceCount}</strong>
        </span>
        <span>
          Shared IPs: <strong>{model.sharedIpCount}</strong>
        </span>
      </div>
    </section>
  )
}

function buildNetworkModel(
  activeTransaction: TransactionFlag,
  transactions: TransactionFlag[],
) {
  const activeCard = activeTransaction.cardId
  const activeDevice = activeTransaction.deviceId
  const activeIp = activeTransaction.ipAddress

  const related = transactions.filter((tx) => {
    if (tx.cardId === activeCard) {
      return true
    }
    if (activeDevice && tx.deviceId === activeDevice) {
      return true
    }
    if (activeIp && tx.ipAddress === activeIp) {
      return true
    }
    return false
  })

  const cardCounts = countValues(related.map((tx) => tx.cardId))
  const deviceCounts = countValues(related.map((tx) => tx.deviceId ?? ''))
  const ipCounts = countValues(related.map((tx) => tx.ipAddress ?? ''))

  const cardKeys = sortedKeys(cardCounts, 8)
  const deviceKeys = sortedKeys(deviceCounts, 6)
  const ipKeys = sortedKeys(ipCounts, 6)

  const cardNodes = layoutColumn(
    cardKeys.map((card) => ({
      id: `card:${card}`,
      key: card,
      label: card,
      type: 'card' as const,
      linkedCount: cardCounts.get(card) ?? 1,
      isActive: card === activeCard,
    })),
    100,
  )
  const deviceNodes = layoutColumn(
    deviceKeys.map((device) => ({
      id: `device:${device}`,
      key: device,
      label: shortLabel(device, 11),
      type: 'device' as const,
      linkedCount: deviceCounts.get(device) ?? 1,
      isActive: Boolean(activeDevice) && device === activeDevice,
    })),
    380,
  )
  const ipNodes = layoutColumn(
    ipKeys.map((ip) => ({
      id: `ip:${ip}`,
      key: ip,
      label: shortLabel(ip, 12),
      type: 'ip' as const,
      linkedCount: ipCounts.get(ip) ?? 1,
      isActive: Boolean(activeIp) && ip === activeIp,
    })),
    650,
  )

  const nodes = [...cardNodes, ...deviceNodes, ...ipNodes]
  const nodeById = new Map(nodes.map((node) => [node.id, node]))

  const edgeWeights = new Map<string, number>()
  const activeEdgeIds = new Set<string>()
  const nodeTransactionIds = new Map<string, Set<string>>()

  const linkNode = (nodeId: string, transactionId: string) => {
    const current = nodeTransactionIds.get(nodeId) ?? new Set<string>()
    current.add(transactionId)
    nodeTransactionIds.set(nodeId, current)
  }

  for (const tx of related) {
    const cardId = `card:${tx.cardId}`
    linkNode(cardId, tx.transactionId)
    if (tx.deviceId && deviceCounts.has(tx.deviceId)) {
      const deviceId = `device:${tx.deviceId}`
      linkNode(deviceId, tx.transactionId)
      const edgeId = `${cardId}->${deviceId}`
      edgeWeights.set(edgeId, (edgeWeights.get(edgeId) ?? 0) + 1)

      if (tx.transactionId === activeTransaction.transactionId) {
        activeEdgeIds.add(edgeId)
      }

      if (tx.ipAddress && ipCounts.has(tx.ipAddress)) {
        const ipId = `ip:${tx.ipAddress}`
        linkNode(ipId, tx.transactionId)
        const edgeTwoId = `${deviceId}->${ipId}`
        edgeWeights.set(edgeTwoId, (edgeWeights.get(edgeTwoId) ?? 0) + 1)

        if (tx.transactionId === activeTransaction.transactionId) {
          activeEdgeIds.add(edgeTwoId)
        }
      }
    } else if (tx.ipAddress && ipCounts.has(tx.ipAddress)) {
      const ipId = `ip:${tx.ipAddress}`
      linkNode(ipId, tx.transactionId)
      const edgeId = `${cardId}->${ipId}`
      edgeWeights.set(edgeId, (edgeWeights.get(edgeId) ?? 0) + 1)

      if (tx.transactionId === activeTransaction.transactionId) {
        activeEdgeIds.add(edgeId)
      }
    }
  }

  const edges: NetworkEdge[] = Array.from(edgeWeights.entries()).map(
    ([edgeId, weight]) => {
      const [from, to] = edgeId.split('->')
      return {
        id: edgeId,
        from,
        to,
        isActive: activeEdgeIds.has(edgeId),
        weight,
      }
    },
  )

  return {
    edges,
    nodeById,
    nodeTransactionIds: new Map(
      Array.from(nodeTransactionIds.entries()).map(([key, value]) => [
        key,
        Array.from(value),
      ]),
    ),
    nodes,
    relatedCardCount: cardKeys.length,
    sharedDeviceCount: deviceKeys.length,
    sharedIpCount: ipKeys.length,
  }
}

function layoutColumn(
  nodes: Array<Omit<NetworkNode, 'x' | 'y'>>,
  x: number,
): NetworkNode[] {
  if (nodes.length === 0) {
    return []
  }

  const step = HEIGHT / (nodes.length + 1)
  return nodes.map((node, index) => ({
    ...node,
    x,
    y: Math.round(step * (index + 1)),
  }))
}

function countValues(values: string[]) {
  const map = new Map<string, number>()
  for (const value of values) {
    const key = value.trim()
    if (!key) {
      continue
    }

    map.set(key, (map.get(key) ?? 0) + 1)
  }

  return map
}

function sortedKeys(map: Map<string, number>, max: number) {
  return Array.from(map.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, max)
    .map(([key]) => key)
}

function shortLabel(value: string, max: number) {
  if (value.length <= max) {
    return value
  }

  return `${value.slice(0, max - 2)}..`
}
