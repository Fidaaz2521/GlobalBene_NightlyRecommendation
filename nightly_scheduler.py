# nightly_scheduler.py
import logging
import time
import math
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    FAISS_INDEX_PATH,
    EMBEDDINGS_PATH,
    TOP_K,
    CACHE_EXPIRY_HOURS,
    SBERT_CACHE_DIR,
)
from database import get_mongo_connection, get_all_users
from embedding_generator import get_embedding_generator
from faiss_indexer import FAISSIndexer, get_faiss_indexer
from upstash_client import upstash_client

logger = logging.getLogger("nightly_scheduler")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


def build_post_embeddings_and_index(batch_size: int = 256) -> Dict[str, Any]:
    """
    - Fetch all posts from Mongo
    - Build embeddings in batches (to avoid memory spike)
    - Build FAISS index and save to disk
    - Save a small mapping file (post_ids) via the FAISS indexer.save_index()
    Returns dict with stats.
    """
    conn = get_mongo_connection()
    posts_df = conn.get_posts()

    if posts_df is None or posts_df.empty:
        logger.warning("No posts found in DB. Aborting build.")
        return {"status": "no_posts"}

    embedder = get_embedding_generator()

    # ensure post_id present
    if "post_id" not in posts_df.columns:
        posts_df = posts_df.rename(columns={"_id": "post_id"})
    post_ids = posts_df["post_id"].astype(str).tolist()

    n = len(posts_df)
    logger.info(f"Fetched {n} posts. Building embeddings in batches (batch_size={batch_size})")

    # accumulate embeddings in float32
    embeddings_list = []
    for i in tqdm(range(0, n, batch_size), desc="Embedding batches", unit="batch"):
        batch_df = posts_df.iloc[i : i + batch_size]
        texts = []
        for _, post in batch_df.iterrows():
            caption = str(post.get("caption", "")) if "caption" in post else ""
            body = str(post.get("body", "")) if "body" in post else ""
            title = str(post.get("title", "")) if "title" in post else ""
            texts.append(f"{caption} {title} {body}".strip())

        # encode
        emb = embedder.model.encode(texts, convert_to_numpy=True)
        if emb.ndim == 1:
            emb = np.expand_dims(emb, 0)
        embeddings_list.append(emb.astype("float32"))

    embeddings = np.vstack(embeddings_list).astype("float32")

    # create FAISS index (cosine via normalized IP)
    indexer = FAISSIndexer()
    indexer.create_index_cosine(embeddings, post_ids)
    # save index + ids
    idx_path = Path(FAISS_INDEX_PATH)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    indexer.save_index(str(idx_path))

    # optionally save embeddings to disk (for future incremental updates)
    emb_path = Path(EMBEDDINGS_PATH)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pickle
        with open(emb_path, "wb") as f:
            pickle.dump({"embeddings": embeddings, "post_ids": post_ids}, f)
        logger.info(f"Saved embeddings to {emb_path}")
    except Exception as e:
        logger.exception(f"Failed saving embeddings file: {e}")

    return {"status": "ok", "n_posts": n, "index_path": str(idx_path), "embeddings_path": str(emb_path)}


def compute_and_store_topk_for_all_users(k: int = TOP_K, batch_size: int = 128) -> Dict[str, int]:
    """
    For each user:
      - generate user embedding (SBERT)
      - query FAISS index (k)
      - format recommendations
      - store to upstash using upstash_client.store_user_recommendations
    """
    users = get_all_users()
    if not users:
        logger.warning("No users returned by get_all_users(). Nothing to compute.")
        return {"processed": 0, "stored": 0, "failed": 0}

    embedder = get_embedding_generator()
    indexer = get_faiss_indexer()

    processed = 0
    stored = 0
    failed = 0

    logger.info(f"Starting Top-K generation for {len(users)} users (k={k})")

    # process users in batches to avoid large loops causing connection issues
    for i in range(0, len(users), batch_size):
        batch = users[i : i + batch_size]
        texts = []
        user_ids = []
        for u in batch:
            uid = str(u.get("user_id") or u.get("_id"))
            user_ids.append(uid)
            username = str(u.get("username", "")) if u.get("username") else ""
            bio = str(u.get("bio", "")) if u.get("bio") else ""
            interests = str(u.get("interests", "")) if u.get("interests") else ""
            texts.append(f"{username} {bio} {interests}".strip() or "user")

        # encode all users in this batch
        try:
            user_embeddings = embedder.model.encode(texts, convert_to_numpy=True)
            if user_embeddings.ndim == 1:
                user_embeddings = np.expand_dims(user_embeddings, 0)
        except Exception as e:
            logger.exception(f"Embedding generation failed for user batch: {e}")
            failed += len(batch)
            continue

               # query FAISS for each user in the batch
        for uid, emb in zip(user_ids, user_embeddings):
            try:
                distances, item_ids = indexer.search(emb, k=k)
                # distances shape (1,k) or (k,)
                # convert to scores: higher = better
                dists = distances[0] if distances.ndim == 2 else distances
                scores = [float(1.0 / (1.0 + float(d))) for d in dists]

                recs = []
                # item_ids is already a 1D list of post_ids
                for rank, (pid, sc) in enumerate(zip(item_ids, scores), start=1):
                    recs.append({"item_id": str(pid), "score": sc, "rank": rank})


                # store in upstash (with TTL from config)
                ok = upstash_client.store_user_recommendations(uid, recs, expiry_hours=CACHE_EXPIRY_HOURS)
                if ok:
                    stored += 1
                else:
                    failed += 1

                processed += 1
            except Exception as e:
                logger.exception(f"Failed computing Top-K for user {uid}: {e}")
                failed += 1

    logger.info(f"Completed Top-K: processed={processed}, stored={stored}, failed={failed}")
    return {"processed": processed, "stored": stored, "failed": failed}


def run_full_nightly_job():
    start = time.time()
    logger.info("=== NIGHTLY JOB START ===")

    try:
        # 1) Build or refresh FAISS index from posts (rebuild each night).
        build_stats = build_post_embeddings_and_index(batch_size=256)
        logger.info(f"Index build stats: {build_stats}")

        # 2) Compute top-K per user and store to Upstash
        stats = compute_and_store_topk_for_all_users(k=TOP_K, batch_size=128)
        logger.info(f"Top-K stats: {stats}")

    except Exception as e:
        logger.exception(f"Nightly job failed: {e}")
    finally:
        elapsed = time.time() - start
        logger.info(f"=== NIGHTLY JOB FINISHED in {elapsed:.1f}s ===")


if __name__ == "__main__":
    run_full_nightly_job()

