from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.api.routes import router as api_router
from app.core.settings import settings
from app.utils.logger import logger

app = FastAPI(
    title="Javanese Hate Speech Detection API",
    description="API untuk deteksi ujaran kebencian dalam bahasa Jawa",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if they exist
static_dir = os.path.join(os.path.dirname(__file__), "api", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include API routes
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Javanese Hate Speech Detection API...")
    logger.info(f"API running on {settings.api_host}:{settings.api_port}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Javanese Hate Speech Detection API...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)