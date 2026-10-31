import { useState, useCallback } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import type { Capacity, PredictionInput, PredictionOutput, HistoryEntry } from '@/schemas/types'
import { predictionSchema, type PredictionFormData } from '@/schemas/validation'
import { buildSchema, getExampleData } from '@/lib/schema'
import { api } from '@/lib/api'
import { CapacitySelector } from '@/components/ui/CapacitySelector'
import { StepIndicator } from '@/components/ui/StepIndicator'
import { NumberField, SelectField } from '@/components/ui/FormFields'
import { ResultPanel } from './ResultPanel'
import { HistoryPanel } from './HistoryPanel'
import { AboutModel } from './AboutModel'
import { useHistory } from '@/hooks/useHistory'
import { clsx } from 'clsx'

type Status = 'idle' | 'loading' | 'success' | 'error'

const STEP_LABELS = ['Profil Rumah', 'Info Token', 'Perangkat', 'Konfirmasi']

export function PredictionPage() {
  const [capacity, setCapacity] = useState<Capacity>('900')
  const [step, setStep] = useState(0)
  const [status, setStatus] = useState<Status>('idle')
  const [result, setResult] = useState<PredictionOutput | null>(null)
  const [errorMsg, setErrorMsg] = useState('')
  const [copyFeedback, setCopyFeedback] = useState(false)

  const history = useHistory()
  const schema = buildSchema(capacity)

  const form = useForm<PredictionFormData>({
    resolver: zodResolver(predictionSchema),
    defaultValues: {
      kota: '',
      jumlah_anggota: 4,
      status_subsidi: '',
      nominal_token: 100000,
      frekuensi_isi: 2,
      kulkas_jumlah: 1,
      kulkas_kategori: 'Kecil / 1 pintu',
      tv_jumlah: 1,
      tv_kategori: 'Sedang, sekitar 2-5 jam per hari',
      ac_jumlah: 0,
      ac_pk: 'Tidak ada AC',
      ac_kategori: 'Tidak ada / tidak digunakan',
      kipas_jumlah: 2,
      kipas_kategori: 'Sedang, sekitar 2-5 jam per hari',
      rice_cooker_jumlah: 1,
      rice_cooker_kategori: 'Sedang, sekitar 2-5 jam per hari',
      mesin_cuci_jumlah: 1,
      mesin_cuci_kategori: 'Sedang, 3-4 kali per minggu',
    },
    mode: 'onBlur',
  })

  const { register, handleSubmit, formState: { errors }, reset, setValue, watch, trigger } = form

  const watchAll = watch()

  const handleCapacityChange = useCallback(
    (c: Capacity) => {
      setCapacity(c)
      setStep(0)
      setResult(null)
      setStatus('idle')
      // Keep values but reset AC defaults for 900
      if (c === '900') {
        setValue('ac_jumlah', 0)
        setValue('ac_pk', 'Tidak ada AC')
        setValue('ac_kategori', 'Tidak ada / tidak digunakan')
      }
    },
    [setValue],
  )

  const fillExample = useCallback(() => {
    const example = getExampleData(capacity)
    Object.entries(example).forEach(([key, val]) => {
      setValue(key as keyof PredictionFormData, val as string | number)
    })
  }, [capacity, setValue])

  const handleReset = useCallback(() => {
    reset()
    setResult(null)
    setStatus('idle')
    setStep(0)
  }, [reset])

  const handleReload = useCallback(
    (entry: HistoryEntry) => {
      const inp = entry.input
      setCapacity(inp.capacity)
      Object.entries(inp).forEach(([key, val]) => {
        if (key !== 'capacity') {
          setValue(key as keyof PredictionFormData, val as string | number)
        }
      })
      setStep(3)
      setResult(null)
      setStatus('idle')
    },
    [setValue],
  )

  const onSubmit = async (data: PredictionFormData) => {
    setStatus('loading')
    setErrorMsg('')
    try {
      const input: PredictionInput = { capacity, ...data }
      const res = await api.predict(input)
      setResult(res)
      setStatus('success')
      history.add(input, res)
    } catch (err) {
      setStatus('error')
      setErrorMsg(err instanceof Error ? err.message : 'Terjadi kesalahan saat menghubungi server.')
    }
  }

  const handleCopy = useCallback(() => {
    if (!result) return
    const text = `Prediksi Token Listrik\nDaya: ${result.capacity} VA\nDurasi: ${result.predicted_days.toFixed(1)} hari\nModel: ${result.model_version}`
    navigator.clipboard.writeText(text).then(() => {
      setCopyFeedback(true)
      setTimeout(() => setCopyFeedback(false), 2000)
    })
  }, [result])

  const canGoNext = async () => {
    const fieldsPerStep: (keyof PredictionFormData)[][] = [
      ['kota', 'jumlah_anggota', 'status_subsidi'],
      ['nominal_token', 'frekuensi_isi'],
      [
        'kulkas_jumlah', 'kulkas_kategori',
        'tv_jumlah', 'tv_kategori',
        'ac_jumlah', 'ac_pk', 'ac_kategori',
        'kipas_jumlah', 'kipas_kategori',
        'rice_cooker_jumlah', 'rice_cooker_kategori',
        'mesin_cuci_jumlah', 'mesin_cuci_kategori',
      ],
      [],
    ]
    const fieldsToValidate = fieldsPerStep[step]
    if (fieldsToValidate.length === 0) return true
    return trigger(fieldsToValidate)
  }

  const nextStep = async () => {
    const valid = await canGoNext()
    if (valid && step < 3) setStep(step + 1)
  }

  const prevStep = () => {
    if (step > 0) setStep(step - 1)
  }

  // Conditional visibility for appliance category fields
  const showField = (fieldName: string): boolean => {
    const deps: Record<string, { field: keyof PredictionFormData; hideValue: number }> = {
      kulkas_kategori: { field: 'kulkas_jumlah', hideValue: 0 },
      tv_kategori: { field: 'tv_jumlah', hideValue: 0 },
      ac_pk: { field: 'ac_jumlah', hideValue: 0 },
      ac_kategori: { field: 'ac_jumlah', hideValue: 0 },
      kipas_kategori: { field: 'kipas_jumlah', hideValue: 0 },
      rice_cooker_kategori: { field: 'rice_cooker_jumlah', hideValue: 0 },
      mesin_cuci_kategori: { field: 'mesin_cuci_jumlah', hideValue: 0 },
    }
    const dep = deps[fieldName]
    if (!dep) return true
    return Number(watchAll[dep.field]) !== dep.hideValue
  }

  return (
    <main className="mx-auto max-w-3xl px-4 py-6 pb-16">
      {/* Capacity selection sits above the form steps */}
      <div className="mb-6">
        <CapacitySelector value={capacity} onChange={handleCapacityChange} />
      </div>

      <StepIndicator steps={STEP_LABELS} current={step} />

      <form
        onSubmit={handleSubmit(onSubmit)}
        noValidate
        className="rounded-xl border border-slate-200 dark:border-slate-700 bg-surface-light dark:bg-surface-dark-card p-5 sm:p-6"
      >
        {/* Step header */}
        <div className="mb-5">
          <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200">
            {schema.sections[Math.min(step, schema.sections.length - 1)]?.title ?? 'Konfirmasi'}
          </h2>
          {step < schema.sections.length && schema.sections[step]?.description && (
            <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
              {schema.sections[step].description}
            </p>
          )}
        </div>

        {/* Step 0: Profil */}
        <div className={clsx(step !== 0 && 'hidden')} role="group" aria-label="Profil Rumah Tangga">
          <div className="space-y-4">
            <SelectField
              label="Kota / Kabupaten"
              options={schema.sections[0].fields.find((f) => f.name === 'kota')!.options!}
              placeholder="Pilih kota..."
              helpText="Lokasi rumah Anda. Mempengaruhi tarif pajak penerangan jalan (PBJT)."
              error={errors.kota?.message}
              {...register('kota')}
            />
            <NumberField
              label="Jumlah Anggota Keluarga"
              unit="orang"
              min={1}
              max={20}
              helpText="Total orang yang tinggal di rumah ini."
              error={errors.jumlah_anggota?.message}
              {...register('jumlah_anggota')}
            />
            <SelectField
              label="Status Subsidi Listrik"
              options={['Subsidi', 'Non Subsidi']}
              placeholder="Pilih status..."
              helpText={
                capacity === '900'
                  ? 'Pelanggan 900 VA umumnya mendapat subsidi.'
                  : 'Pelanggan 1300 VA ke atas umumnya non-subsidi.'
              }
              error={errors.status_subsidi?.message}
              {...register('status_subsidi')}
            />
          </div>
        </div>

        {/* Step 1: Token */}
        <div className={clsx(step !== 1 && 'hidden')} role="group" aria-label="Informasi Token">
          <div className="space-y-4">
            <NumberField
              label="Nominal Token"
              unit="Rp"
              min={20000}
              max={5000000}
              helpText="Nilai token yang akan Anda beli (contoh: 50.000, 100.000, 200.000)."
              error={errors.nominal_token?.message}
              {...register('nominal_token')}
            />
            <NumberField
              label="Frekuensi Isi Token"
              unit="kali per bulan"
              min={1}
              max={30}
              helpText="Berapa kali biasanya Anda mengisi ulang token dalam sebulan."
              error={errors.frekuensi_isi?.message}
              {...register('frekuensi_isi')}
            />
          </div>
        </div>

        {/* Step 2: Perangkat */}
        <div className={clsx(step !== 2 && 'hidden')} role="group" aria-label="Perangkat Listrik">
          <div className="space-y-6">
            {/* Kulkas */}
            <fieldset className="space-y-3">
              <legend className="text-sm font-medium text-slate-600 dark:text-slate-400">Kulkas</legend>
              <NumberField label="Jumlah" unit="unit" min={0} max={5} error={errors.kulkas_jumlah?.message} {...register('kulkas_jumlah')} />
              {showField('kulkas_kategori') && (
                <SelectField
                  label="Ukuran Kulkas"
                  options={['Tidak ada', 'Kecil / 1 pintu', 'Sedang / 2 pintu', 'Besar / side by side', 'Tidak tahu']}
                  helpText="Ukuran kulkas yang paling sering digunakan."
                  error={errors.kulkas_kategori?.message}
                  {...register('kulkas_kategori')}
                />
              )}
            </fieldset>

            {/* TV */}
            <fieldset className="space-y-3">
              <legend className="text-sm font-medium text-slate-600 dark:text-slate-400">TV</legend>
              <NumberField label="Jumlah" unit="unit" min={0} max={5} error={errors.tv_jumlah?.message} {...register('tv_jumlah')} />
              {showField('tv_kategori') && (
                <SelectField
                  label="Penggunaan"
                  options={['Tidak ada / tidak digunakan', 'Jarang, kurang dari 2 jam per hari', 'Sedang, sekitar 2-5 jam per hari', 'Sering, sekitar 6-10 jam per hari', 'Sangat sering, lebih dari 10 jam per hari']}
                  helpText="Rata-rata berapa lama TV menyala per hari."
                  error={errors.tv_kategori?.message}
                  {...register('tv_kategori')}
                />
              )}
            </fieldset>

            {/* AC */}
            <fieldset className="space-y-3">
              <legend className="text-sm font-medium text-slate-600 dark:text-slate-400">AC</legend>
              <NumberField label="Jumlah" unit="unit" min={0} max={5} error={errors.ac_jumlah?.message} {...register('ac_jumlah')} />
              {showField('ac_pk') && (
                <>
                  <SelectField
                    label="Kapasitas (PK)"
                    options={['Tidak ada AC', '1/2 PK', '3/4 PK', '1 PK', '1.5 PK', '2 PK atau lebih', 'Tidak tahu']}
                    helpText="PK (Paardekracht) menunjukkan kapasitas pendinginan AC."
                    error={errors.ac_pk?.message}
                    {...register('ac_pk')}
                  />
                  <SelectField
                    label="Penggunaan"
                    options={['Tidak ada / tidak digunakan', 'Jarang, kurang dari 2 jam per hari', 'Sedang, sekitar 2-5 jam per hari', 'Sering, sekitar 6-10 jam per hari', 'Sangat sering, lebih dari 10 jam per hari']}
                    helpText="Rata-rata berapa lama AC menyala per hari."
                    error={errors.ac_kategori?.message}
                    {...register('ac_kategori')}
                  />
                </>
              )}
            </fieldset>

            {/* Kipas */}
            <fieldset className="space-y-3">
              <legend className="text-sm font-medium text-slate-600 dark:text-slate-400">Kipas Angin</legend>
              <NumberField label="Jumlah" unit="unit" min={0} max={10} error={errors.kipas_jumlah?.message} {...register('kipas_jumlah')} />
              {showField('kipas_kategori') && (
                <SelectField
                  label="Penggunaan"
                  options={['Tidak ada / tidak digunakan', 'Jarang, kurang dari 2 jam per hari', 'Sedang, sekitar 2-5 jam per hari', 'Sering, sekitar 6-10 jam per hari', 'Sangat sering, lebih dari 10 jam per hari']}
                  helpText="Rata-rata berapa lama kipas menyala per hari."
                  error={errors.kipas_kategori?.message}
                  {...register('kipas_kategori')}
                />
              )}
            </fieldset>

            {/* Rice Cooker */}
            <fieldset className="space-y-3">
              <legend className="text-sm font-medium text-slate-600 dark:text-slate-400">Rice Cooker</legend>
              <NumberField label="Jumlah" unit="unit" min={0} max={3} error={errors.rice_cooker_jumlah?.message} {...register('rice_cooker_jumlah')} />
              {showField('rice_cooker_kategori') && (
                <SelectField
                  label="Penggunaan"
                  options={['Tidak ada / tidak digunakan', 'Jarang, kurang dari 2 jam per hari', 'Sedang, sekitar 2-5 jam per hari', 'Sering, sekitar 6-10 jam per hari', 'Sangat sering, lebih dari 10 jam per hari']}
                  helpText="Termasuk mode warm. Rata-rata pemakaian per hari."
                  error={errors.rice_cooker_kategori?.message}
                  {...register('rice_cooker_kategori')}
                />
              )}
            </fieldset>

            {/* Mesin Cuci */}
            <fieldset className="space-y-3">
              <legend className="text-sm font-medium text-slate-600 dark:text-slate-400">Mesin Cuci</legend>
              <NumberField label="Jumlah" unit="unit" min={0} max={3} error={errors.mesin_cuci_jumlah?.message} {...register('mesin_cuci_jumlah')} />
              {showField('mesin_cuci_kategori') && (
                <SelectField
                  label="Penggunaan"
                  options={['Tidak ada / tidak digunakan', 'Jarang, 1-2 kali per minggu', 'Sedang, 3-4 kali per minggu', 'Sering, 5-6 kali per minggu', 'Sangat sering, hampir setiap hari']}
                  helpText="Berapa kali mesin cuci digunakan per minggu."
                  error={errors.mesin_cuci_kategori?.message}
                  {...register('mesin_cuci_kategori')}
                />
              )}
            </fieldset>
          </div>
        </div>

        {/* Step 3: Konfirmasi */}
        <div className={clsx(step !== 3 && 'hidden')} role="group" aria-label="Konfirmasi">
          <div className="space-y-3 text-sm">
            <h3 className="font-medium text-slate-700 dark:text-slate-300">Ringkasan Data Anda</h3>
            <div className="rounded-lg bg-surface-muted dark:bg-surface-dark-elevated p-4 space-y-2">
              <SummaryRow label="Daya" value={`${capacity} VA`} />
              <SummaryRow label="Kota" value={watchAll.kota || '-'} />
              <SummaryRow label="Anggota Keluarga" value={`${watchAll.jumlah_anggota} orang`} />
              <SummaryRow label="Status" value={watchAll.status_subsidi || '-'} />
              <SummaryRow label="Nominal Token" value={`Rp ${Number(watchAll.nominal_token).toLocaleString('id-ID')}`} />
              <SummaryRow label="Frekuensi Isi" value={`${watchAll.frekuensi_isi}x per bulan`} />
              <div className="border-t border-slate-200 dark:border-slate-600 pt-2 mt-2">
                <p className="text-xs text-slate-500 dark:text-slate-400 font-medium mb-1">Perangkat:</p>
                {Number(watchAll.kulkas_jumlah) > 0 && <SummaryRow label="Kulkas" value={`${watchAll.kulkas_jumlah} unit, ${watchAll.kulkas_kategori}`} />}
                {Number(watchAll.tv_jumlah) > 0 && <SummaryRow label="TV" value={`${watchAll.tv_jumlah} unit`} />}
                {Number(watchAll.ac_jumlah) > 0 && <SummaryRow label="AC" value={`${watchAll.ac_jumlah} unit, ${watchAll.ac_pk}`} />}
                {Number(watchAll.kipas_jumlah) > 0 && <SummaryRow label="Kipas" value={`${watchAll.kipas_jumlah} unit`} />}
                {Number(watchAll.rice_cooker_jumlah) > 0 && <SummaryRow label="Rice Cooker" value={`${watchAll.rice_cooker_jumlah} unit`} />}
                {Number(watchAll.mesin_cuci_jumlah) > 0 && <SummaryRow label="Mesin Cuci" value={`${watchAll.mesin_cuci_jumlah} unit`} />}
              </div>
            </div>
          </div>
        </div>

        {/* Error state */}
        {status === 'error' && (
          <div className="mt-4 rounded-lg border border-semantic-error/30 bg-red-50 dark:bg-red-900/20 p-4" role="alert">
            <p className="text-sm text-semantic-error font-medium">Gagal melakukan prediksi</p>
            <p className="text-xs text-semantic-error/80 mt-1">{errorMsg}</p>
          </div>
        )}

        {/* Navigation */}
        <div className="mt-6 flex items-center justify-between gap-3">
          <div className="flex gap-2">
            {step > 0 && (
              <button
                type="button"
                onClick={prevStep}
                className="px-4 py-2 text-sm font-medium rounded-lg border border-slate-300 dark:border-slate-600
                           text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
              >
                Kembali
              </button>
            )}
            <button
              type="button"
              onClick={fillExample}
              className="px-3 py-2 text-xs rounded-lg border border-slate-300 dark:border-slate-600
                         text-slate-500 dark:text-slate-400 hover:border-primary hover:text-primary transition-colors"
            >
              Isi contoh
            </button>
          </div>

          <div className="flex gap-2">
            {step < 3 && (
              <button
                type="button"
                onClick={handleReset}
                className="px-3 py-2 text-sm text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
              >
                Reset
              </button>
            )}
            {step < 3 ? (
              <button
                type="button"
                onClick={nextStep}
                className="px-5 py-2 text-sm font-medium rounded-lg bg-primary text-white
                           hover:bg-primary-700 active:bg-primary-800 transition-colors"
              >
                Lanjut
              </button>
            ) : (
              <button
                type="submit"
                disabled={status === 'loading'}
                className={clsx(
                  'px-6 py-2.5 text-sm font-semibold rounded-lg transition-colors',
                  status === 'loading'
                    ? 'bg-primary/60 text-white/80 cursor-wait'
                    : 'bg-primary text-white hover:bg-primary-700 active:bg-primary-800',
                )}
              >
                {status === 'loading' ? (
                  <span className="flex items-center gap-2">
                    <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
                      <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="opacity-75" />
                    </svg>
                    Memproses...
                  </span>
                ) : (
                  'Prediksi Sekarang'
                )}
              </button>
            )}
          </div>
        </div>
      </form>

      {/* Result */}
      {status === 'success' && result && (
        <div className="mt-6">
          <ResultPanel result={result} onCopy={handleCopy} copyFeedback={copyFeedback} />
        </div>
      )}

      {/* History */}
      <HistoryPanel
        entries={history.entries}
        onReload={handleReload}
        onClear={history.clear}
        compareIds={history.compareIds}
        onToggleCompare={history.toggleCompare}
        compareEntries={history.compareEntries}
      />

      {/* About */}
      <AboutModel />

      {/* Footer */}
      <footer className="mt-8 pt-4 border-t border-slate-200 dark:border-slate-700 text-center">
        <p className="text-xs text-slate-400 dark:text-slate-500">
          Tugas Akhir, Teknik Informatika · Model prediksi berbasis neural network
        </p>
      </footer>
    </main>
  )
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-sm">
      <span className="text-slate-500 dark:text-slate-400">{label}</span>
      <span className="text-slate-800 dark:text-slate-200 font-medium text-right">{value}</span>
    </div>
  )
}
