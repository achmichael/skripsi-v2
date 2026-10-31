from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.app.api.routes import router
from backend.app.config import settings
from backend.app.models.loader import preload_all_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    print("Loading models...")
    preload_all_models()
    print("Models loaded successfully")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Prediksi Token Listrik API",
    description="Neural network model untuk prediksi durasi token listrik prabayar PLN",
    version=settings.model_version,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "Prediksi Token Listrik API",
        "version": settings.model_version,
        "docs": "/docs",
        "health": "/api/health"
    }
