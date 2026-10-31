import { z } from 'zod'

export const predictionSchema = z.object({
  kota: z.string().min(1, 'Pilih kota atau kabupaten'),
  jumlah_anggota: z.coerce
    .number({ invalid_type_error: 'Masukkan angka' })
    .int('Harus bilangan bulat')
    .min(1, 'Minimal 1 orang')
    .max(20, 'Maksimal 20 orang'),
  status_subsidi: z.string().min(1, 'Pilih status subsidi'),
  nominal_token: z.coerce
    .number({ invalid_type_error: 'Masukkan angka' })
    .int('Harus bilangan bulat')
    .min(20000, 'Minimal Rp 20.000')
    .max(5000000, 'Maksimal Rp 5.000.000'),
  frekuensi_isi: z.coerce
    .number({ invalid_type_error: 'Masukkan angka' })
    .int('Harus bilangan bulat')
    .min(1, 'Minimal 1 kali')
    .max(30, 'Maksimal 30 kali'),
  kulkas_jumlah: z.coerce.number().int().min(0).max(5).default(0),
  kulkas_kategori: z.string().default('Tidak ada'),
  tv_jumlah: z.coerce.number().int().min(0).max(5).default(0),
  tv_kategori: z.string().default('Tidak ada / tidak digunakan'),
  ac_jumlah: z.coerce.number().int().min(0).max(5).default(0),
  ac_pk: z.string().default('Tidak ada AC'),
  ac_kategori: z.string().default('Tidak ada / tidak digunakan'),
  kipas_jumlah: z.coerce.number().int().min(0).max(10).default(0),
  kipas_kategori: z.string().default('Tidak ada / tidak digunakan'),
  rice_cooker_jumlah: z.coerce.number().int().min(0).max(3).default(0),
  rice_cooker_kategori: z.string().default('Tidak ada / tidak digunakan'),
  mesin_cuci_jumlah: z.coerce.number().int().min(0).max(3).default(0),
  mesin_cuci_kategori: z.string().default('Tidak ada / tidak digunakan'),
})

export type PredictionFormData = z.infer<typeof predictionSchema>
