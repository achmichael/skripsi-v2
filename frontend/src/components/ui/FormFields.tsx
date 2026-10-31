import { forwardRef } from 'react'
import { clsx } from 'clsx'

interface FormFieldProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string
  error?: string
  helpText?: string
  unit?: string
}

export const NumberField = forwardRef<HTMLInputElement, FormFieldProps>(
  ({ label, error, helpText, unit, id, className, ...props }, ref) => {
    const fieldId = id || props.name || ''
    const errorId = `${fieldId}-error`
    const helpId = `${fieldId}-help`

    return (
      <div className={clsx('space-y-1', className)}>
        <label htmlFor={fieldId} className="block text-sm font-medium text-slate-700 dark:text-slate-300">
          {label}
          {unit && <span className="ml-1 text-xs text-slate-400 font-normal">({unit})</span>}
        </label>
        <input
          ref={ref}
          id={fieldId}
          type="number"
          inputMode="numeric"
          aria-invalid={!!error}
          aria-describedby={clsx(error && errorId, helpText && helpId) || undefined}
          className={clsx(
            'w-full rounded-md border px-3 py-2 text-sm bg-surface-light dark:bg-surface-dark-elevated',
            'text-slate-900 dark:text-slate-100 placeholder:text-slate-400',
            'transition-colors',
            'hover:border-slate-400 dark:hover:border-slate-500',
            error
              ? 'border-semantic-error focus:ring-semantic-error'
              : 'border-slate-300 dark:border-slate-600 focus:ring-primary focus:border-primary',
          )}
          {...props}
        />
        {helpText && !error && (
          <p id={helpId} className="text-xs text-slate-500 dark:text-slate-400">
            {helpText}
          </p>
        )}
        {error && (
          <p id={errorId} className="text-xs text-semantic-error" role="alert">
            {error}
          </p>
        )}
      </div>
    )
  },
)
NumberField.displayName = 'NumberField'

interface SelectFieldProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label: string
  options: string[]
  error?: string
  helpText?: string
  placeholder?: string
}

export const SelectField = forwardRef<HTMLSelectElement, SelectFieldProps>(
  ({ label, options, error, helpText, placeholder, id, className, ...props }, ref) => {
    const fieldId = id || props.name || ''
    const errorId = `${fieldId}-error`
    const helpId = `${fieldId}-help`

    return (
      <div className={clsx('space-y-1', className)}>
        <label htmlFor={fieldId} className="block text-sm font-medium text-slate-700 dark:text-slate-300">
          {label}
        </label>
        <select
          ref={ref}
          id={fieldId}
          aria-invalid={!!error}
          aria-describedby={clsx(error && errorId, helpText && helpId) || undefined}
          className={clsx(
            'w-full rounded-md border px-3 py-2 text-sm bg-surface-light dark:bg-surface-dark-elevated',
            'text-slate-900 dark:text-slate-100',
            'transition-colors',
            'hover:border-slate-400 dark:hover:border-slate-500',
            error
              ? 'border-semantic-error focus:ring-semantic-error'
              : 'border-slate-300 dark:border-slate-600 focus:ring-primary focus:border-primary',
          )}
          {...props}
        >
          {placeholder && (
            <option value="" disabled>
              {placeholder}
            </option>
          )}
          {options.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
        {helpText && !error && (
          <p id={helpId} className="text-xs text-slate-500 dark:text-slate-400">
            {helpText}
          </p>
        )}
        {error && (
          <p id={errorId} className="text-xs text-semantic-error" role="alert">
            {error}
          </p>
        )}
      </div>
    )
  },
)
SelectField.displayName = 'SelectField'
