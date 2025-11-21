import logging
import json

from celery_app import app  # ⬅️ use the configured Celery app instead of shared_task

from database import get_all_users, get_user_data
from model_loader import get_recommendation_model
from embedding_generator import get_embedding_generator
from faiss_indexer import get_faiss_indexer
from upstash_client import upstash_client
from config import TOP_K, CACHE_EXPIRY_HOURS

logger = logging.getLogger(__name__)


@app.task(bind=True, max_retries=3, default_retry_delay=60, name="tasks.generate_recommendations_task")
def generate_recommendations_task(self):
    """
    Nightly batch: Generate and directly cache recommendations in Upstash for all users.
    """
    try:
        logger.info("🌙 Starting nightly batch recommendation generation...")
        recommendation_model = get_recommendation_model()
        embedding_generator = get_embedding_generator()
        faiss_indexer = get_faiss_indexer()

        users = get_all_users()
        logger.info(f"Processing {len(users)} users...")

        user_recommendations = {}

        for idx, user in enumerate(users):
            user_id = str(user.get("user_id"))

            profile_text = (
                str(user.get("username", "")) + " "
                + str(user.get("bio", "")) + " "
                + str(user.get("interests", ""))
            )

            embedding = embedding_generator.model.encode(
                [profile_text], convert_to_numpy=True
            )

            distances, item_ids = faiss_indexer.search(embedding[0], k=TOP_K)

            recommendations = [
                {
                    "item_id": str(item_id),
                    "score": float(1 / (1 + distance)),
                    "rank": rank + 1,
                }
                for rank, (distance, item_id) in enumerate(
                    zip(distances[0], item_ids[0])
                )
            ]

            # Store in Upstash Redis, per-user
            upstash_client.store_user_recommendations(
                user_id, recommendations, expiry_hours=CACHE_EXPIRY_HOURS
            )
            user_recommendations[user_id] = recommendations

            if (idx + 1) % 100 == 0:
                logger.info(f"✓ Done for {idx + 1}/{len(users)} users")

        logger.info(f"✅ Nightly batch complete for {len(users)} users")
        return {
            "status": "success",
            "users_processed": len(users),
        }

    except Exception as exc:
        logger.error(f"❌ Task failed: {str(exc)}")
        raise self.retry(exc=exc, countdown=60)


@app.task(name="tasks.refresh_single_user_recommendations")
def refresh_single_user_recommendations(user_id: str):
    """
    On-demand: Generate and cache recommendations in Upstash for a single user.
    """
    try:
        logger.info(f"🔄 Refreshing recommendations for user {user_id}...")
        recommendation_model = get_recommendation_model()
        embedding_generator = get_embedding_generator()
        faiss_indexer = get_faiss_indexer()

        user = get_user_data(user_id)
        if not user:
            logger.warning(f"No user found with ID {user_id}")
            return {"status": "not_found", "user_id": user_id}

        profile_text = (
            str(user.get("username", "")) + " "
            + str(user.get("bio", "")) + " "
            + str(user.get("interests", ""))
        )

        embedding = embedding_generator.model.encode(
            [profile_text], convert_to_numpy=True
        )

        distances, item_ids = faiss_indexer.search(embedding[0], k=TOP_K)

        recommendations = [
            {
                "item_id": str(item_id),
                "score": float(1 / (1 + distance)),
                "rank": rank + 1,
            }
            for rank, (distance, item_id) in enumerate(
                zip(distances[0], item_ids[0])
            )
        ]

        # Store in Upstash Redis for this user
        upstash_client.store_user_recommendations(
            user_id, recommendations, expiry_hours=CACHE_EXPIRY_HOURS
        )

        return {
            "status": "success",
            "user_id": user_id,
            "recommendations": recommendations,
        }

    except Exception as e:
        logger.error(f"Error refreshing for user {user_id}: {str(e)}")
        return {"status": "failed", "error": str(e)}
