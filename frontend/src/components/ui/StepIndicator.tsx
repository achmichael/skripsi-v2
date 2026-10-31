import { clsx } from 'clsx'

interface StepIndicatorProps {
  steps: string[]
  current: number
}

export function StepIndicator({ steps, current }: StepIndicatorProps) {
  return (
    <nav aria-label="Langkah formulir" className="mb-6">
      <ol className="flex items-center gap-1">
        {steps.map((label, i) => {
          const done = i < current
          const active = i === current
          return (
            <li key={label} className="flex items-center gap-1 flex-1">
              <div className="flex flex-col items-center flex-1 min-w-0">
                <div
                  className={clsx(
                    'w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold border-2 transition-colors',
                    done && 'bg-primary border-primary text-white',
                    active && 'border-primary text-primary bg-primary-50 dark:bg-primary-900/30',
                    !done && !active && 'border-slate-300 dark:border-slate-600 text-slate-400 dark:text-slate-500',
                  )}
                  aria-current={active ? 'step' : undefined}
                >
                  {done ? (
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                         strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  ) : (
                    i + 1
                  )}
                </div>
                <span
                  className={clsx(
                    'text-[11px] mt-1 text-center leading-tight hidden sm:block truncate max-w-full',
                    active ? 'text-primary font-medium dark:text-primary-light' : 'text-slate-400 dark:text-slate-500',
                  )}
                >
                  {label}
                </span>
              </div>
              {i < steps.length - 1 && (
                <div
                  className={clsx(
                    'h-0.5 flex-1 rounded-full min-w-3',
                    done ? 'bg-primary' : 'bg-slate-200 dark:bg-slate-700',
                  )}
                  aria-hidden="true"
                />
              )}
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
