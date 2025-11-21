# main_api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
from tasks import refresh_single_user_recommendations
from topk_hybrid_advanced import get_recommender

app = FastAPI()
logger = logging.getLogger(__name__)

recommender = None


@app.on_event("startup")
async def startup_event():
    global recommender
    recommender = get_recommender()
    logger.info("Recommender initialized on startup")


@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str, background_tasks: BackgroundTasks):
    """
    Flow:
      1) If cache hit -> return cache
      2) If cold-start -> compute heuristics + collaborative synchronously -> return + cache
      3) Else -> enqueue Celery on-demand refresh and return 202 'generating'
    """
    try:
        result = recommender.get_hybrid_recommendations(user_id)

        # If recommendations present (cache or cold_start) -> return
        if result.get("recommendations") is not None:
            logger.info(f"Returning {result.get('source')} recommendations for {user_id}")
            return JSONResponse({
                "user_id": user_id,
                "recommendations": result["recommendations"],
                "source": result.get("source"),
                "strategy": result.get("strategy"),
            })

        # Otherwise: cache miss and not cold-start -> enqueue celery
        logger.info(f"Cache miss & not cold-start for {user_id} -> enqueueing Celery task")
        refresh_single_user_recommendations.apply_async(args=[user_id], queue="recommendations")

        return JSONResponse(
            status_code=202,
            content={
                "user_id": user_id,
                "status": "generating",
                "message": "Recommendations are being generated in background. Please retry in a few seconds.",
                "source": "on-demand"
            }
        )

    except Exception as e:
        logger.exception(f"Error in recommendations endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommendations/refresh/{user_id}")
async def manual_refresh(user_id: str):
    try:
        result = refresh_single_user_recommendations.apply_async(args=[user_id], queue="recommendations")
        return {"status": "refresh_queued", "user_id": user_id, "task_id": result.id}
    except Exception as e:
        logger.exception("Failed to enqueue refresh task")
        raise HTTPException(status_code=500, detail=str(e))
