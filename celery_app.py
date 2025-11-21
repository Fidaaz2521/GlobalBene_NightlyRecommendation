from celery import Celery
from celery.schedules import crontab
import os
import ssl
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Read broker & backend URLs from environment
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
backend_url = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Debug prints (optional – remove later if you want)
print("CELERY_BROKER_URL =", broker_url)
print("CELERY_RESULT_BACKEND =", backend_url)


app = Celery(
    "globalbene_automation",
    include=["tasks"],   
)



# Base Celery config
app.conf.update(
    broker_url=broker_url,
    result_backend=backend_url,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Celery Beat Schedule - Runs at 2:00 AM daily
    beat_schedule={
        "generate-recommendations-nightly": {
            "task": "tasks.generate_recommendations_task",
            "schedule": crontab(hour=2, minute=0),  # 2:00 AM every day
            "options": {"queue": "recommendations"},
        },
    },
)

# If using rediss:// (TLS with Upstash), configure SSL
if broker_url.startswith("rediss://"):
    app.conf.broker_use_ssl = {
        "ssl_cert_reqs": ssl.CERT_NONE,  # simple and works with Upstash
    }

if backend_url.startswith("rediss://"):
    app.conf.redis_backend_use_ssl = {
        "ssl_cert_reqs": ssl.CERT_NONE,
    }


if __name__ == "__main__":
    app.start()
