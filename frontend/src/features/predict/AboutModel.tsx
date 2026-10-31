export function AboutModel() {
  return (
    <section className="mt-8 rounded-lg border border-slate-200 dark:border-slate-700 bg-surface-light dark:bg-surface-dark-card p-5">
      <h2 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
        Tentang Model
      </h2>
      <dl className="space-y-3 text-sm">
        <div>
          <dt className="text-xs text-slate-500 dark:text-slate-400">Judul Penelitian</dt>
          <dd className="text-slate-800 dark:text-slate-200 mt-0.5 leading-relaxed">
            Prediksi Masa Pakai Token Listrik Rumah Tangga Menggunakan Neural Network
          </dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500 dark:text-slate-400">Jenis Model</dt>
          <dd className="text-slate-800 dark:text-slate-200 mt-0.5">
            Multilayer Perceptron (MLP), diimplementasikan dari nol menggunakan NumPy
          </dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500 dark:text-slate-400">Model Tersedia</dt>
          <dd className="text-slate-800 dark:text-slate-200 mt-0.5">
            3 model terpisah untuk daya 900 VA, 1300 VA, dan 2200 VA
          </dd>
        </div>
        <div>
          <dt className="text-xs text-slate-500 dark:text-slate-400">Metrik Evaluasi</dt>
          <dd className="text-slate-800 dark:text-slate-200 mt-0.5">
            {/* [REAL DATA] — isi dengan metrik evaluasi dari hasil training */}
            <span className="text-xs italic text-slate-400">[Akan diisi dengan data evaluasi sesungguhnya]</span>
          </dd>
        </div>
      </dl>
      <p className="mt-4 text-xs text-slate-500 dark:text-slate-400 leading-relaxed border-t border-slate-200 dark:border-slate-700 pt-3">
        Model ini dikembangkan untuk keperluan penelitian tugas akhir (skripsi).
        Hasil prediksi bersifat estimasi dan dapat berbeda dari kondisi nyata.
        Data pelatihan dikumpulkan melalui survei terhadap pengguna listrik prabayar PLN.
      </p>
    </section>
  )
}
