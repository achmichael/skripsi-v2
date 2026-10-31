from fastapi import APIRouter, HTTPException
from typing import Dict, List
import sys
import os

# Add parent directory to path (go up 3 levels: routes.py -> api -> app -> backend -> root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.app.schemas.prediction import (
    PredictionInput, 
    PredictionOutput, 
    FormSchema, 
    FieldSchema
)
from backend.app.models.loader import get_model
from backend.app.config import settings

from predict import predict
from preprocessing.preprocess import handle_missing_value
from preprocessing.compute_derived_columns import compute_derived_columns
from utils.helper import frequency_encode, frequency_mapping, ordinal_encode
from feature_extraction.feature_extraction import extractor
from core.config.config import configs
import pandas as pd

router = APIRouter(prefix="/api")

# City options for dropdown
CITIES = [
    "Bandung", "Bekasi", "Bogor", "Cimahi", "Cirebon", "Depok",
    "Sukabumi", "Tasikmalaya", "Banjar", "Bandung Barat", "Ciamis",
    "Cianjur", "Garut", "Indramayu", "Karawang", "Kuningan",
    "Majalengka", "Pangandaran", "Purwakarta", "Subang", "Sumedang",
    "Tasikmalaya (Kab)", "Lainnya"
]

APPLIANCE_CATEGORIES = {
    "kulkas": ["Tidak ada", "Jarang (< 12 jam/hari)", "Sedang (12-20 jam/hari)", "Sering (> 20 jam/hari)"],
    "tv": ["Tidak ada / tidak digunakan", "Jarang (< 4 jam/hari)", "Sedang (4-8 jam/hari)", "Sering (> 8 jam/hari)"],
    "ac": ["Tidak ada / tidak digunakan", "Jarang (< 4 jam/hari)", "Sedang (4-8 jam/hari)", "Sering (> 8 jam/hari)"],
    "kipas": ["Tidak ada / tidak digunakan", "Jarang (< 4 jam/hari)", "Sedang (4-8 jam/hari)", "Sering (> 8 jam/hari)"],
    "rice_cooker": ["Tidak ada / tidak digunakan", "Jarang (< 2 jam/hari)", "Sedang (2-4 jam/hari)", "Sering (> 4 jam/hari)"],
    "mesin_cuci": ["Tidak ada / tidak digunakan", "Jarang (< 2 jam/minggu)", "Sedang (2-5 jam/minggu)", "Sering (> 5 jam/minggu)"]
}

AC_PK_OPTIONS = ["Tidak ada AC", "0.5 PK", "0.75 PK", "1 PK", "1.5 PK", "2 PK"]

@router.options("/health")
def health_options():
    return {}

@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_version": settings.model_version,
        "capacities_available": ["900", "1300", "2200"]
    }

@router.get("/schema", response_model=FormSchema)
def get_form_schema(capacity: str):
    """Return dynamic form schema for selected capacity."""
    if capacity not in ["900", "1300", "2200"]:
        raise HTTPException(400, "Capacity harus 900, 1300, atau 2200")
    
    fields = [
        FieldSchema(
            name="kota",
            label="Kota/Kabupaten",
            type="select",
            options=CITIES,
            required=True,
            help_text="Pilih lokasi rumah Anda"
        ),
        FieldSchema(
            name="jumlah_anggota",
            label="Jumlah Anggota Keluarga",
            type="number",
            unit="orang",
            min=1,
            max=20,
            required=True,
            help_text="Total anggota keluarga yang tinggal di rumah"
        ),
        FieldSchema(
            name="status_subsidi",
            label="Status Subsidi Listrik",
            type="select",
            options=["Subsidi", "Non Subsidi"],
            required=True,
            help_text="Status subsidi listrik PLN Anda"
        ),
        FieldSchema(
            name="nominal_token",
            label="Nominal Token",
            type="number",
            unit="Rp",
            min=20000,
            max=5000000,
            required=True,
            help_text="Nilai token yang akan dibeli (contoh: 50000 untuk Rp 50.000)"
        ),
        FieldSchema(
            name="frekuensi_isi",
            label="Frekuensi Isi Token",
            type="number",
            unit="kali/bulan",
            min=1,
            max=30,
            required=True,
            help_text="Berapa kali Anda mengisi token dalam sebulan"
        ),
    ]
    
    return FormSchema(capacity=capacity, fields=fields)

@router.get("/appliance-categories")
def get_appliance_categories():
    """Return appliance category options."""
    return {
        "kulkas": APPLIANCE_CATEGORIES["kulkas"],
        "tv": APPLIANCE_CATEGORIES["tv"],
        "ac": APPLIANCE_CATEGORIES["ac"],
        "ac_pk": AC_PK_OPTIONS,
        "kipas": APPLIANCE_CATEGORIES["kipas"],
        "rice_cooker": APPLIANCE_CATEGORIES["rice_cooker"],
        "mesin_cuci": APPLIANCE_CATEGORIES["mesin_cuci"]
    }

@router.options("/predict")
def predict_options():
    return {}

@router.post("/predict", response_model=PredictionOutput)
def predict_token_lifespan(input_data: PredictionInput):
    """Run prediction using selected capacity model."""
    try:
        # Build DataFrame from input
        df_input = pd.DataFrame([{
            "Kota/Kabupaten": input_data.kota,
            "Daya_Listrik_Rumah_VA": int(input_data.capacity),
            "Jumlah_Anggota_Keluarga": input_data.jumlah_anggota,
            "Status_Subsidi_Listrik": input_data.status_subsidi,
            "Nominal_Token_Terakhir_Rp": input_data.nominal_token,
            "Frekuensi_Isi_Token_Per_Bulan": input_data.frekuensi_isi,
            "Kulkas_Jumlah": input_data.kulkas_jumlah,
            "Kulkas_Kategori": input_data.kulkas_kategori,
            "TV_Jumlah": input_data.tv_jumlah,
            "TV_Kategori": input_data.tv_kategori,
            "AC_Jumlah": input_data.ac_jumlah,
            "AC_PK_Kategori": input_data.ac_pk,
            "AC_Kategori": input_data.ac_kategori,
            "Kipas_Jumlah": input_data.kipas_jumlah,
            "Kipas_Kategori": input_data.kipas_kategori,
            "RiceCooker_Jumlah": input_data.rice_cooker_jumlah,
            "RiceCooker_Kategori": input_data.rice_cooker_kategori,
            "MesinCuci_Jumlah": input_data.mesin_cuci_jumlah,
            "MesinCuci_Kategori": input_data.mesin_cuci_kategori,
            "Token_Habis_Dalam_Hari": 0
        }])
        
        # Feature engineering pipeline
        handled = handle_missing_value(df_input)
        mapped = frequency_encode(handled, frequency_mapping(df_input, configs))
        computed = compute_derived_columns(mapped)
        encoded = ordinal_encode(computed, configs)
        
        # Extract features
        x_data, _, _, _ = extractor(encoded, input_data.capacity)
        
        # Load model and predict
        model, scalers = get_model(input_data.capacity)
        cfg = configs["models_config"][input_data.capacity]
        
        prediction = predict(
            model=model,
            x_data=x_data,
            x_scaler=scalers['x_scaler'],
            y_scaler=scalers['y_scaler'],
            use_log=cfg['use_log_transform']
        )
        
        result = round(float(prediction[0]), 2)
        
        return PredictionOutput(
            predicted_days=result,
            capacity=input_data.capacity,
            model_version=settings.model_version
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Kesalahan prediksi: {str(e)}")
