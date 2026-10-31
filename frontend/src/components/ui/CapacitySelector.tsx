import type { Capacity } from '@/schemas/types'
import { clsx } from 'clsx'

interface CapacitySelectorProps {
  value: Capacity
  onChange: (c: Capacity) => void
}

const CAPACITIES: { value: Capacity; label: string; desc: string }[] = [
  { value: '900', label: '900 VA', desc: 'Rumah tangga kecil' },
  { value: '1300', label: '1300 VA', desc: 'Rumah tangga menengah' },
  { value: '2200', label: '2200 VA', desc: 'Rumah tangga besar' },
]

export function CapacitySelector({ value, onChange }: CapacitySelectorProps) {
  return (
    <fieldset>
      <legend className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
        Daya Listrik Rumah
      </legend>
      <div className="grid grid-cols-3 gap-2 sm:gap-3" role="radiogroup" aria-label="Pilih daya listrik">
        {CAPACITIES.map((cap) => {
          const selected = value === cap.value
          return (
            <button
              key={cap.value}
              type="button"
              role="radio"
              aria-checked={selected}
              onClick={() => onChange(cap.value)}
              className={clsx(
                'relative rounded-lg border-2 px-3 py-3 text-center transition-colors',
                'hover:border-primary-light focus-visible:ring-2 focus-visible:ring-primary',
                selected
                  ? 'border-primary bg-primary-50 dark:bg-primary-900/30 dark:border-primary-light'
                  : 'border-slate-200 dark:border-slate-600 bg-surface-light dark:bg-surface-dark-card',
              )}
            >
              <span
                className={clsx(
                  'block text-lg font-semibold',
                  selected ? 'text-primary dark:text-primary-light' : 'text-slate-800 dark:text-slate-200',
                )}
              >
                {cap.label}
              </span>
              <span className="block text-xs text-slate-500 dark:text-slate-400 mt-0.5">{cap.desc}</span>
            </button>
          )
        })}
      </div>
    </fieldset>
  )
}
