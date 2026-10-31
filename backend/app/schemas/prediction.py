from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional

class Appliance(BaseModel):
    name: str = Field(..., description="Nama perangkat")
    jumlah: int = Field(..., ge=0, le=10, description="Jumlah unit")
    kategori: str = Field(..., description="Kategori penggunaan")

class PredictionInput(BaseModel):
    capacity: Literal["900", "1300", "2200"] = Field(..., description="Daya listrik rumah (VA)")
    kota: str = Field(..., description="Kota/Kabupaten")
    jumlah_anggota: int = Field(..., ge=1, le=20, description="Jumlah anggota keluarga")
    status_subsidi: Literal["Subsidi", "Non Subsidi"] = Field(..., description="Status subsidi listrik")
    nominal_token: int = Field(..., ge=20000, le=5000000, description="Nominal token (Rp)")
    frekuensi_isi: int = Field(..., ge=1, le=30, description="Frekuensi isi token per bulan")
    
    # Appliances
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

class PredictionOutput(BaseModel):
    predicted_days: float = Field(..., description="Prediksi durasi token (hari)")
    capacity: str = Field(..., description="Daya yang dipilih")
    model_version: str = Field(..., description="Versi model")

class FieldSchema(BaseModel):
    name: str
    label: str
    type: str
    unit: Optional[str] = None
    min: Optional[int | float] = None
    max: Optional[int | float] = None
    required: bool = True
    help_text: Optional[str] = None
    options: Optional[List[str]] = None
    default: Optional[str | int] = None

class FormSchema(BaseModel):
    capacity: str
    fields: List[FieldSchema]
