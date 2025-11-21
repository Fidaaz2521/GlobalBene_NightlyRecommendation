import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# ============================================================================
# UPSTASH REDIS CONFIGURATION (Cloud-Based - Primary)
# ============================================================================
UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")
UPSTASH_REDIS_URL = os.getenv("UPSTASH_REDIS_URL")  # redis://:token@host:port


# ============================================================================
# CELERY CONFIGURATION (Uses Upstash Cloud Redis)
# ============================================================================
CELERY_BROKER_URL = UPSTASH_REDIS_URL or os.getenv(
    "CELERY_BROKER_URL", 
    "redis://localhost:6379/0"  # Fallback for local development only
)
CELERY_RESULT_BACKEND = UPSTASH_REDIS_URL or os.getenv(
    "CELERY_RESULT_BACKEND",
    "redis://localhost:6379/0"  # Fallback for local development only
)

CELERY_CONFIG = {
    "broker_url": CELERY_BROKER_URL,
    "result_backend": CELERY_RESULT_BACKEND,
    "task_serializer": "json",
    "accept_content": ["json"],
    "result_serializer": "json",
    "timezone": "UTC",
    "enable_utc": True,
    "task_track_started": True,
    "task_time_limit": 30 * 60,  # 30 minutes
    "task_soft_time_limit": 25 * 60,  # 25 minutes
}


# ============================================================================
# PROJECT DIRECTORIES
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# ============================================================================
# MONGODB CONFIGURATION
# ============================================================================
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "global_bene")


# ============================================================================
# LOCAL REDIS CONFIGURATION (Fallback - Not used if Upstash is set)
# ============================================================================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_TTL = 24 * 3600  # 24 hours in seconds


# ============================================================================
# SBERT MODEL CONFIGURATION
# ============================================================================
SBERT_MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")
SBERT_CACHE_DIR = os.getenv(
    "SBERT_CACHE_DIR", 
    str(MODELS_DIR / "sbert_model")
)


# ============================================================================
# FAISS INDEX & EMBEDDINGS CONFIGURATION
# ============================================================================
FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH", 
    str(MODELS_DIR / "faiss_index.bin")
)
EMBEDDINGS_PATH = os.getenv(
    "EMBEDDINGS_PATH", 
    str(MODELS_DIR / "post_embeddings.pkl")
)


# ============================================================================
# RECOMMENDATION MODEL CONFIGURATION
# ============================================================================
MODEL_PATH = os.getenv(
    "MODEL_PATH", 
    str(MODELS_DIR / "recommendation_model.pkl")
)
TOP_K = int(os.getenv("TOP_K", 50))
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))
CACHE_EXPIRY_HOURS = int(os.getenv("CACHE_EXPIRY_HOURS", 24))


# ============================================================================
# FASTAPI CONFIGURATION
# ============================================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_TITLE = os.getenv("API_TITLE", "GlobalBene Recommendation Engine")
API_VERSION = os.getenv("API_VERSION", "1.0.0")


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = os.getenv("LOG_DIR", str(LOGS_DIR))
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ============================================================================
# VALIDATION: Ensure Critical Environment Variables Are Set
# ============================================================================
if not UPSTASH_REDIS_URL:
    print("  WARNING: UPSTASH_REDIS_URL not set. Using local Redis fallback.")
    print("   For production, set UPSTASH_REDIS_URL in .env file")

if not MONGO_URI:
    raise ValueError(" MONGO_URI is required. Set it in .env file.")

if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
    print("  WARNING: Upstash REST credentials not set. Recommendations storage may fail.")
