import type { HistoryEntry } from '@/schemas/types'
import { clsx } from 'clsx'

interface HistoryPanelProps {
  entries: HistoryEntry[]
  onReload: (entry: HistoryEntry) => void
  onClear: () => void
  compareIds: [string, string] | null
  onToggleCompare: (id: string) => void
  compareEntries: HistoryEntry[]
}

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' })
}

function formatRp(n: number): string {
  return 'Rp ' + n.toLocaleString('id-ID')
}

export function HistoryPanel({
  entries,
  onReload,
  onClear,
  compareIds,
  onToggleCompare,
  compareEntries,
}: HistoryPanelProps) {
  if (entries.length === 0) return null

  return (
    <section className="mt-8">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300">
          Riwayat Prediksi ({entries.length})
        </h2>
        <button
          onClick={onClear}
          className="text-xs text-slate-500 hover:text-semantic-error dark:text-slate-400 transition-colors"
        >
          Hapus semua
        </button>
      </div>

      {entries.length >= 2 && (
        <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">
          Pilih 2 prediksi untuk membandingkan.
        </p>
      )}

      <ul className="space-y-2" role="list">
        {entries.map((entry) => {
          const isComparing = compareIds?.includes(entry.id)
          return (
            <li
              key={entry.id}
              className={clsx(
                'rounded-lg border p-3 transition-colors',
                isComparing
                  ? 'border-accent bg-accent/5 dark:border-accent-dark'
                  : 'border-slate-200 dark:border-slate-700 bg-surface-light dark:bg-surface-dark-card',
              )}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <p className="text-sm font-medium text-slate-800 dark:text-slate-200">
                    {entry.result.predicted_days.toFixed(1)} hari
                    <span className="text-xs text-slate-500 dark:text-slate-400 ml-2">
                      {entry.input.capacity} VA
                    </span>
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400 truncate">
                    {formatRp(entry.input.nominal_token)} · {entry.input.kota} · {formatTime(entry.timestamp)}
                  </p>
                </div>
                <div className="flex gap-1.5 flex-shrink-0">
                  {entries.length >= 2 && (
                    <button
                      onClick={() => onToggleCompare(entry.id)}
                      className={clsx(
                        'text-xs px-2 py-1 rounded border transition-colors',
                        isComparing
                          ? 'border-accent text-accent-dark dark:text-accent-light bg-accent/10'
                          : 'border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-400 hover:border-accent',
                      )}
                    >
                      {isComparing ? 'Batal' : 'Bandingkan'}
                    </button>
                  )}
                  <button
                    onClick={() => onReload(entry)}
                    className="text-xs px-2 py-1 rounded border border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-400 hover:border-primary hover:text-primary transition-colors"
                  >
                    Muat ulang
                  </button>
                </div>
              </div>
            </li>
          )
        })}
      </ul>

      {compareEntries.length === 2 && (
        <div className="mt-4 rounded-lg border border-accent/40 bg-accent/5 p-4" aria-live="polite">
          <h3 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">Perbandingan</h3>
          <div className="grid grid-cols-2 gap-4 text-sm">
            {compareEntries.map((e, i) => (
              <div key={e.id}>
                <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">Prediksi {i + 1}</p>
                <p className="text-2xl font-bold font-mono text-primary dark:text-primary-light">
                  {e.result.predicted_days.toFixed(1)}
                  <span className="text-sm font-normal text-slate-500 ml-1">hari</span>
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  {e.input.capacity} VA · {formatRp(e.input.nominal_token)}
                </p>
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-3">
            Selisih:{' '}
            <span className="font-medium text-slate-700 dark:text-slate-300">
              {Math.abs(compareEntries[0].result.predicted_days - compareEntries[1].result.predicted_days).toFixed(1)} hari
            </span>
          </p>
        </div>
      )}
    </section>
  )
}
