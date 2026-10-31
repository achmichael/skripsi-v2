import { clsx } from 'clsx'
import { useTheme } from '@/hooks/useTheme'

interface HeaderProps {
  backendStatus: 'checking' | 'online' | 'offline'
}

export function Header({ backendStatus }: HeaderProps) {
  const { theme, toggle } = useTheme()

  const statusColor =
    backendStatus === 'online'
      ? 'bg-semantic-success'
      : backendStatus === 'offline'
        ? 'bg-semantic-error'
        : 'bg-slate-400 animate-pulse'

  const statusLabel =
    backendStatus === 'online' ? 'Terhubung' : backendStatus === 'offline' ? 'Tidak terhubung' : 'Memeriksa...'

  return (
    <header className="border-b border-slate-200 dark:border-slate-700 bg-surface-light dark:bg-surface-dark-card">
      <div className="mx-auto max-w-3xl px-4 py-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <div
            className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center flex-shrink-0"
            aria-hidden="true"
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path
                d="M11.5 2L4 11h5.5l-1 7L16 9h-5.5l1-7z"
                stroke="#fff"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <div className="min-w-0">
            <h1 className="text-lg font-semibold text-slate-900 dark:text-slate-100 truncate leading-tight">
              Prediksi Token Listrik
            </h1>
            <p className="text-xs text-slate-500 dark:text-slate-400 truncate">
              Estimasi durasi token prabayar PLN
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 flex-shrink-0">
          <div className="flex items-center gap-1.5" aria-label={`Status backend: ${statusLabel}`}>
            <span className={clsx('w-2 h-2 rounded-full', statusColor)} aria-hidden="true" />
            <span className="text-xs text-slate-500 dark:text-slate-400 hidden sm:inline">{statusLabel}</span>
          </div>

          <button
            onClick={toggle}
            aria-label={theme === 'light' ? 'Aktifkan mode gelap' : 'Aktifkan mode terang'}
            className="w-9 h-9 rounded-lg border border-slate-200 dark:border-slate-600 flex items-center justify-center
                       text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700
                       transition-colors"
          >
            {theme === 'light' ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                   strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M21 12.79A9 9 0 1111.21 3a7 7 0 009.79 9.79z" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                   strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="5" />
                <line x1="12" y1="1" x2="12" y2="3" /><line x1="12" y1="21" x2="12" y2="23" />
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" /><line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                <line x1="1" y1="12" x2="3" y2="12" /><line x1="21" y1="12" x2="23" y2="12" />
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" /><line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </header>
  )
}
