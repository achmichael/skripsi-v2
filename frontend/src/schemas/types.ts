export type Capacity = '900' | '1300' | '2200'

export interface FieldSchema {
  name: string
  label: string
  type: 'number' | 'select' | 'radio'
  unit?: string
  min?: number
  max?: number
  required: boolean
  help_text?: string
  options?: string[]
  default_value?: string | number
  section: string
  depends_on?: { field: string; not_value: string }
}

export interface FormSchema {
  capacity: Capacity
  sections: SectionSchema[]
}

export interface SectionSchema {
  id: string
  title: string
  description?: string
  fields: FieldSchema[]
}

export interface PredictionInput {
  capacity: Capacity
  kota: string
  jumlah_anggota: number
  status_subsidi: string
  nominal_token: number
  frekuensi_isi: number
  kulkas_jumlah: number
  kulkas_kategori: string
  tv_jumlah: number
  tv_kategori: string
  ac_jumlah: number
  ac_pk: string
  ac_kategori: string
  kipas_jumlah: number
  kipas_kategori: string
  rice_cooker_jumlah: number
  rice_cooker_kategori: string
  mesin_cuci_jumlah: number
  mesin_cuci_kategori: string
}

export interface PredictionOutput {
  predicted_days: number
  capacity: string
  model_version: string
}

export interface HealthResponse {
  status: string
  model_version: string
  capacities_available: string[]
}

export interface HistoryEntry {
  id: string
  timestamp: number
  input: PredictionInput
  result: PredictionOutput
}
