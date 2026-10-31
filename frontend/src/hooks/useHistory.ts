import { useState, useCallback } from 'react'
import type { HistoryEntry, PredictionInput, PredictionOutput } from '@/schemas/types'

let idCounter = 0

export function useHistory() {
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [compareIds, setCompareIds] = useState<[string, string] | null>(null)

  const add = useCallback((input: PredictionInput, result: PredictionOutput) => {
    const entry: HistoryEntry = {
      id: `pred-${++idCounter}-${Date.now()}`,
      timestamp: Date.now(),
      input,
      result,
    }
    setEntries((prev) => [entry, ...prev])
    return entry
  }, [])

  const clear = useCallback(() => {
    setEntries([])
    setCompareIds(null)
  }, [])

  const toggleCompare = useCallback((id: string) => {
    setCompareIds((prev) => {
      if (!prev) return [id, '']
      if (prev[0] === id) return null
      if (prev[1] === id) return [prev[0], '']
      if (!prev[1]) return [prev[0], id]
      return [id, '']
    })
  }, [])

  const compareEntries = compareIds
    ? ([
        entries.find((e) => e.id === compareIds[0]),
        entries.find((e) => e.id === compareIds[1]),
      ].filter(Boolean) as HistoryEntry[])
    : []

  return { entries, add, clear, toggleCompare, compareIds, compareEntries }
}
