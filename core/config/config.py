from typing import Any

PROFILE = [
    "Jumlah_Anggota_Keluarga",
    "Status_Subsidi_Listrik",
]

TOKEN_BEHAVIOR = [
    "Nominal_Token_Terakhir_Rp",
    "Frekuensi_Isi_Token_Per_Bulan",
]

KULKAS = [
    "Kulkas_Jumlah", "Kulkas_Kategori", "Kulkas_EstimasiWattPerUnit",
    "Kulkas_EstimasiJamPerHari", "Kulkas_Energi_kWhPerHari",
]

TV = [
    "TV_Jumlah", "TV_Kategori", "TV_EstimasiWattPerUnit",
    "TV_EstimasiJamPerHari", "TV_Energi_kWhPerHari",
]

AC = [
    "AC_Jumlah", "AC_PK_Kategori", "AC_Kategori", "AC_EstimasiWattPerUnit",
    "AC_EstimasiJamPerHari", "AC_Energi_kWhPerHari",
]

KIPAS = [
    "Kipas_Jumlah", "Kipas_Kategori", "Kipas_EstimasiWattPerUnit",
    "Kipas_EstimasiJamPerHari", "Kipas_Energi_kWhPerHari",
]

RICE_COOKER = [
    "RiceCooker_Jumlah", "RiceCooker_Kategori", "RiceCooker_EstimasiWattPerUnit",
    "RiceCooker_EstimasiJamPerHari", "RiceCooker_Energi_kWhPerHari",
]

MESIN_CUCI = [
    "MesinCuci_Jumlah", "MesinCuci_Kategori", "MesinCuci_EstimasiWattPerUnit",
    "MesinCuci_EstimasiFrekuensiPerMinggu", "MesinCuci_EstimasiDurasiSekaliPakaiJam",
    "MesinCuci_Energi_kWhPerHari",
]

REDUNDANT_WITH_ENERGY = [
    "Kulkas_EstimasiWattPerUnit", "Kulkas_EstimasiJamPerHari",
    "TV_EstimasiWattPerUnit", "TV_EstimasiJamPerHari",
    "AC_EstimasiWattPerUnit", "AC_EstimasiJamPerHari",
    "Kipas_EstimasiWattPerUnit", "Kipas_EstimasiJamPerHari",
    "RiceCooker_EstimasiWattPerUnit", "RiceCooker_EstimasiJamPerHari",
    "MesinCuci_EstimasiWattPerUnit", "MesinCuci_EstimasiFrekuensiPerMinggu",
    "MesinCuci_EstimasiDurasiSekaliPakaiJam",
]

FEATURES = (
    PROFILE + TOKEN_BEHAVIOR + KULKAS + TV + AC + KIPAS + RICE_COOKER + MESIN_CUCI
)


COMMON = {
    "layer_sizes": None,
    "min_delta": 1e-5,
    "lr_decay": 0.001,
    "clip_value": 5.0,
    "batch_size": 16,
    "target_label": "durasi token (hari)",
    "target": "Token_Habis_Dalam_Hari"
}

configs: dict[str, Any] = {
    "models_config": {
        "900": {
            **COMMON,
            "dataset_path": "data/prabayar_900.csv",
            "model_path": "results/prabayar_900/models/model_prabayar_900.json",
            "metrics_dir": "results/prabayar_900/metrics",
            "hidden_layers": [32, 16],
            "learning_rate": 0.005,
            "patience": 50,
            "l2_lambda": 0.003,
            "use_log_transform": False,
            "asymmetric_alpha": 0.3,
            "asymmetric_beta": 0.7,
            "features": list(FEATURES),
        },
        "1300": {
            **COMMON,
            "dataset_path": "data/prabayar_1300.csv",
            "model_path": "results/prabayar_1300/models/model_prabayar_1300.json",
            "metrics_dir": "results/prabayar_1300/metrics",
            "hidden_layers": [16, 8],
            "learning_rate": 0.003,
            "patience": 60,
            "l2_lambda": 0.01,
            "use_log_transform": True,
            "asymmetric_alpha": 0.6,
            "asymmetric_beta": 0.4,
            "features": list(FEATURES),
        },
        "2200": {
            **COMMON,
            "dataset_path": "data/prabayar_2200.csv",
            "model_path": "results/prabayar_2200/models/model_prabayar_2200.json",
            "metrics_dir": "results/prabayar_2200/metrics",
            "hidden_layers": [12],
            "learning_rate": 0.003,
            "patience": 60,
            "l2_lambda": 0.03,
            "use_log_transform": True,
            "asymmetric_alpha": 0.6,
            "asymmetric_beta": 0.4,
            "features": list(FEATURES),
        }
    },
    "preprocessing": {
        "nullable_cols": [
            "Kulkas_Kategori",
            "Alat_Lain_1_Kategori",
            "Alat_Lain_2_Kategori",
            "Alat_Lain_3_Kategori"
        ],
        "ordinal_encoding": {
            "Kulkas_Kategori": {
                "Tidak ada": 0,
                "Tidak tahu": 0,
                "Kecil / 1 pintu": 1,
                "Sedang / 2 pintu": 2,
                "Besar / side by side": 3,
            },
            "TV_Kategori": {
                "Tidak ada / tidak digunakan": 0,
                "Jarang, kurang dari 2 jam per hari": 1,
                "Sedang, sekitar 2-5 jam per hari": 2,
                "Sering, sekitar 6-10 jam per hari": 3,
                "Sangat sering, lebih dari 10 jam per hari": 4,
            },
            "AC_Kategori": {
                "Tidak ada / tidak digunakan": 0,
                "Jarang, kurang dari 2 jam per hari": 1,
                "Sedang, sekitar 2-5 jam per hari": 2,
                "Sering, sekitar 6-10 jam per hari": 3,
                "Sangat sering, lebih dari 10 jam per hari": 4,
            },
            "Kipas_Kategori": {
                "Tidak ada / tidak digunakan": 0,
                "Jarang, kurang dari 2 jam per hari": 1,
                "Sedang, sekitar 2-5 jam per hari": 2,
                "Sering, sekitar 6-10 jam per hari": 3,
                "Sangat sering, lebih dari 10 jam per hari": 4,
            },
            "RiceCooker_Kategori": {
                "Tidak ada / tidak digunakan": 0,
                "Jarang, kurang dari 2 jam per hari": 1,
                "Sedang, sekitar 2-5 jam per hari": 2,
                "Sering, sekitar 6-10 jam per hari": 3,
                "Sangat sering, lebih dari 10 jam per hari": 4,
            },
            "MesinCuci_Kategori": {
                "Tidak ada / tidak digunakan": 0,
                "Jarang, 1-2 kali per minggu": 1,
                "Sedang, 3-4 kali per minggu": 2,
                "Sering, 5-6 kali per minggu": 3,
                "Sangat sering, hampir setiap hari": 4,
            },
            "Alat_Lain_Ada": {
                "Tidak": 0,
                "Ya": 1,
            },
            "Alat_Lain_1_Kategori": {
                "Tidak diisi": 0,
                "Jarang, 1-2 kali per minggu": 1,
                "Sedang, 3-4 kali per minggu": 2,
                "Sering, hampir setiap hari": 3,
                "Sangat sering, setiap hari dan cukup lama": 4,
            },
            "Alat_Lain_2_Kategori": {
                "Tidak diisi": 0,
                "Jarang, 1-2 kali per minggu": 1,
                "Sedang, 3-4 kali per minggu": 2,
                "Sering, hampir setiap hari": 3,
                "Sangat sering, setiap hari dan cukup lama": 4,
            },
            "Alat_Lain_3_Kategori": {
                "Tidak diisi": 0,
                "Jarang, 1-2 kali per minggu": 1,
                "Sedang, 3-4 kali per minggu": 2,
                "Sering, hampir setiap hari": 3,
                "Sangat sering, setiap hari dan cukup lama": 4,
            },
            "AC_PK_Kategori": {
                "Tidak ada AC": 0,
                "Tidak tahu": 0,
                "1/2 PK": 1,
                "3/4 PK": 2,
                "1 PK": 3,
                "1.5 PK": 4,
                "2 PK": 5,
            },
        },
        "no_scaling_cols": [
            "Kulkas_Kategori",
            "TV_Kategori",
            "AC_Kategori",
            "Kipas_Kategori",
            "RiceCooker_Kategori",
            "MesinCuci_Kategori",
            "Alat_Lain_Ada",
            "Alat_Lain_1_Kategori",
            "Alat_Lain_2_Kategori",
            "Alat_Lain_3_Kategori",
            "AC_PK_Kategori",
            "Status_Subsidi_Listrik",
            "Token_Nominal_Kategori",
            "Alat_Lain_1_Jenis",
            "Alat_Lain_2_Jenis",
            "Alat_Lain_3_Jenis",
        ],
        "missing_value_strategy": ["mode", "IQR", "clear"],
        "frequency_cols": [
            "Alat_Lain_1_Jenis",
            "Alat_Lain_2_Jenis",
            "Alat_Lain_3_Jenis"
        ],
        "frequency_categories": {
            "Alat_Lain_1_Jenis": [
                "Charger HP/perangkat kecil",
                "Blender/Mixer",
                "Setrika",
                "Dispenser",
                "Komputer/Laptop",
                "Pompa air",
                "Oven/Microwave",
                "Lainnya",
            ],
            "Alat_Lain_2_Jenis": [
                "Charger HP/perangkat kecil",
                "Blender/Mixer",
                "Setrika",
                "Dispenser",
                "Komputer/Laptop",
                "Pompa air",
                "Oven/Microwave",
                "Lainnya",
            ],
            "Alat_Lain_3_Jenis": [
                "Charger HP/perangkat kecil",
                "Blender/Mixer",
                "Setrika",
                "Dispenser",
                "Komputer/Laptop",
                "Pompa air",
                "Oven/Microwave",
                "Lainnya",
            ],
        },
    }
}
