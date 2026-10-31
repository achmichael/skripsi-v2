from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np
from typing import Optional
import traceback

from predict import load_model, load_scalers, predict
from preprocessing.preprocess import handle_missing_value
from preprocessing.compute_derived_columns import compute_derived_columns
from utils.helper import frequency_encode, frequency_mapping, ordinal_encode
from feature_extraction.feature_extraction import extractor
from core.config.config import configs
import pandas as pd

app = FastAPI(title="Prediksi Durasi Token Listrik")

models_cache = {}
scalers_cache = {}

def init_models():
    for capacity in ["900", "1300", "2200"]:
        cfg = configs["models_config"][capacity]
        models_cache[capacity] = load_model(cfg['model_path'], capacity)
        scalers_cache[capacity] = load_scalers(cfg['metrics_dir'])

init_models()

class PredictionInput(BaseModel):
    capacity: str = Field(..., description="Daya listrik rumah (900/1300/2200)")
    kota: str = Field(..., description="Kota/Kabupaten")
    jumlah_anggota: int = Field(..., ge=1, le=20)
    status_subsidi: str = Field(..., description="Subsidi atau Non Subsidi")
    nominal_token: int = Field(..., ge=20000, le=5000000)
    frekuensi_isi: int = Field(..., ge=1, le=30)

    kulkas_jumlah: int = Field(0, ge=0, le=10)
    kulkas_kategori: str = "Tidak ada"

    tv_jumlah: int = Field(0, ge=0, le=10)
    tv_kategori: str = "Tidak ada / tidak digunakan"

    ac_jumlah: int = Field(0, ge=0, le=10)
    ac_pk: str = "Tidak ada AC"
    ac_kategori: str = "Tidak ada / tidak digunakan"

    kipas_jumlah: int = Field(0, ge=0, le=10)
    kipas_kategori: str = "Tidak ada / tidak digunakan"

    rice_cooker_jumlah: int = Field(0, ge=0, le=10)
    rice_cooker_kategori: str = "Tidak ada / tidak digunakan"

    mesin_cuci_jumlah: int = Field(0, ge=0, le=10)
    mesin_cuci_kategori: str = "Tidak ada / tidak digunakan"

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def make_prediction(data: PredictionInput):
    try:
        if data.capacity not in ["900", "1300", "2200"]:
            raise HTTPException(400, "Capacity harus 900, 1300, atau 2200")

        df_input = pd.DataFrame([{
            "Kota/Kabupaten": data.kota,
            "Daya_Listrik_Rumah_VA": int(data.capacity),
            "Jumlah_Anggota_Keluarga": data.jumlah_anggota,
            "Status_Subsidi_Listrik": data.status_subsidi,
            "Nominal_Token_Terakhir_Rp": data.nominal_token,
            "Frekuensi_Isi_Token_Per_Bulan": data.frekuensi_isi,
            "Kulkas_Jumlah": data.kulkas_jumlah,
            "Kulkas_Kategori": data.kulkas_kategori,
            "TV_Jumlah": data.tv_jumlah,
            "TV_Kategori": data.tv_kategori,
            "AC_Jumlah": data.ac_jumlah,
            "AC_PK_Kategori": data.ac_pk,
            "AC_Kategori": data.ac_kategori,
            "Kipas_Jumlah": data.kipas_jumlah,
            "Kipas_Kategori": data.kipas_kategori,
            "RiceCooker_Jumlah": data.rice_cooker_jumlah,
            "RiceCooker_Kategori": data.rice_cooker_kategori,
            "MesinCuci_Jumlah": data.mesin_cuci_jumlah,
            "MesinCuci_Kategori": data.mesin_cuci_kategori,
            "Token_Habis_Dalam_Hari": 0
        }])

        handled = handle_missing_value(df_input)
        mapped = frequency_encode(handled, frequency_mapping(df_input, configs))
        computed = compute_derived_columns(mapped)
        encoded = ordinal_encode(computed, configs)

        x_data, _, _, _ = extractor(encoded, data.capacity)

        cfg = configs["models_config"][data.capacity]
        prediction = predict(
            model=models_cache[data.capacity],
            x_data=x_data,
            x_scaler=scalers_cache[data.capacity]['x_scaler'],
            y_scaler=scalers_cache[data.capacity]['y_scaler'],
            use_log=cfg['use_log_transform']
        )

        result = round(prediction[0], 2)

        return JSONResponse({
            "success": True,
            "prediction": result,
            "unit": "hari",
            "capacity": data.capacity
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Prediction error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(models_cache.keys())}

app.mount("/static", StaticFiles(directory="static"), name="static")
