# main_api_hf.py
"""
Lightweight FastAPI app for Hugging Face Space / small host.
Uses topk_hybrid_light.get_recommender.
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from topk_hybrid_light import get_recommender

logger = logging.getLogger("main_api_hf")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

app = FastAPI(title="GlobalBene Light Recommender (HF)")

recommender = None


@app.on_event("startup")
async def startup():
    global recommender
    logger.info("Starting Light HF recommender...")
    recommender = get_recommender()
    logger.info("Recommender ready")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str):
    try:
        result = recommender.get_hybrid_recommendations(user_id)
        # if we have computed recommendations (cache or cold-start) -> return
        if result.get("recommendations") is not None:
            return JSONResponse({
                "user_id": user_id,
                "recommendations": result["recommendations"],
                "source": result.get("source"),
                "strategy": result.get("strategy"),
            })
        # else inform caller that the heavy offline job should produce them
        return JSONResponse(
            status_code=202,
            content={
                "user_id": user_id,
                "status": "no_data",
                "message": "No cached recommendations found and user is not cold-start. Please run nightly precompute or enqueue offline job.",
                "source": result.get("source"),
            },
        )
    except Exception as e:
        logger.exception("Recommendations endpoint error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
