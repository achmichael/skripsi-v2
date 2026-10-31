import type { PredictionOutput } from '@/schemas/types'
import { useEffect, useState } from 'react'

interface ResultPanelProps {
  result: PredictionOutput | null
  onCopy: () => void
  copyFeedback: boolean
}

function formatDate(daysFromNow: number): string {
  const d = new Date()
  d.setDate(d.getDate() + Math.round(daysFromNow))
  return d.toLocaleDateString('id-ID', { day: 'numeric', month: 'long', year: 'numeric' })
}

function CountUp({ target, duration = 1200 }: { target: number; duration?: number }) {
  const [display, setDisplay] = useState(0)

  useEffect(() => {
    const startTime = performance.now()
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    if (prefersReduced) {
      setDisplay(target)
      return
    }

    let raf: number
    const step = (now: number) => {
      const elapsed = now - startTime
      const progress = Math.min(elapsed / duration, 1)
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3)
      setDisplay(Math.round(eased * target * 100) / 100)
      if (progress < 1) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [target, duration])

  return <>{display.toFixed(1)}</>
}

export function ResultPanel({ result, onCopy, copyFeedback }: ResultPanelProps) {
  if (!result) return null

  const days = result.predicted_days
  const endDate = formatDate(days)
  const today = new Date().toLocaleDateString('id-ID', { day: 'numeric', month: 'long', year: 'numeric' })

  return (
    <section
      aria-live="polite"
      aria-atomic="true"
      className="rounded-xl border-2 border-primary/30 bg-primary-50/50 dark:bg-primary-900/20 dark:border-primary/20 p-6"
    >
      <h2 className="text-sm font-medium text-slate-600 dark:text-slate-400 mb-4">
        Hasil Prediksi
      </h2>

      <div className="text-center mb-4">
        <p className="text-5xl sm:text-6xl font-bold font-mono text-primary dark:text-primary-light tracking-tight">
          <CountUp target={days} />
        </p>
        <p className="text-lg font-medium text-slate-700 dark:text-slate-300 mt-1">hari</p>
      </div>

      <div className="bg-surface-light dark:bg-surface-dark-card rounded-lg p-4 space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-slate-500 dark:text-slate-400">Dari hari ini</span>
          <span className="text-slate-800 dark:text-slate-200 font-medium">{today}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-500 dark:text-slate-400">Perkiraan habis</span>
          <span className="text-slate-800 dark:text-slate-200 font-medium">{endDate}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-500 dark:text-slate-400">Daya</span>
          <span className="text-slate-800 dark:text-slate-200 font-medium">{result.capacity} VA</span>
        </div>
        <div className="flex justify-between">
          <span className="text-slate-500 dark:text-slate-400">Model</span>
          <span className="text-slate-800 dark:text-slate-200 font-medium text-xs font-mono">
            {result.model_version}
          </span>
        </div>
      </div>

      <div className="mt-4 flex items-start gap-2">
        <button
          onClick={onCopy}
          className="text-xs text-primary hover:text-primary-light dark:text-primary-light transition-colors underline underline-offset-2"
        >
          {copyFeedback ? 'Tersalin!' : 'Salin hasil'}
        </button>
      </div>

      <p className="mt-3 text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
        Angka di atas adalah estimasi dari model penelitian (neural network). 
        Durasi sesungguhnya dapat berbeda tergantung pola penggunaan listrik harian Anda.
      </p>
    </section>
  )
}
