# topk_hybrid_light.py
"""
Lightweight recommender for low-memory hosts (HuggingFace Space, small VMs).
Does NOT load SBERT / FAISS / PKL. Uses Mongo + votes + Upstash heuristics only.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional

from config import TOP_K
from database import get_mongo_connection
from upstash_client import upstash_client

logger = logging.getLogger(__name__)


class LightTopKRecommender:
    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self.db_conn = get_mongo_connection()
        self.db = self.db_conn.db
        self.cache = upstash_client

        # internal data structures used by heuristics
        self.posts_df: pd.DataFrame = pd.DataFrame()
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.user_post_matrix: Optional[pd.DataFrame] = None
        self.user_similarity_df: Optional[pd.DataFrame] = None

        # initialize from DB
        self._init_heuristics_data()

    # --------------------------- initialization ---------------------------
    def _safe_int(self, value, default: int = 0) -> int:
        import math
        if value is None:
            return default
        try:
            if isinstance(value, float) and math.isnan(value):
                return default
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return default

    def _init_heuristics_data(self):
        logger.info("[LIGHT-H] initializing posts/users/votes from MongoDB...")
        try:
            # load posts
            posts_df = self.db_conn.get_posts()
            if posts_df is None or posts_df.empty:
                logger.info("[LIGHT-H] no posts found")
                self.posts_df = pd.DataFrame()
            else:
                # ensure id columns are strings and basic defaults exist
                if "post_id" in posts_df.columns:
                    posts_df["post_id"] = posts_df["post_id"].astype(str)
                if "community_id" in posts_df.columns:
                    posts_df["community_id"] = posts_df["community_id"].astype(str)
                if "score" not in posts_df.columns:
                    posts_df["score"] = 0
                self.posts_df = posts_df
                logger.info(f"[LIGHT-H] loaded {len(self.posts_df)} posts")

            # load users -> user_profiles
            users_df = self.db_conn.get_users()
            if users_df is None or users_df.empty:
                logger.info("[LIGHT-H] no users found")
                self.user_profiles = {}
            else:
                for _, row in users_df.iterrows():
                    uid = str(row.get("user_id") or row.get("_id") or "")
                    if not uid:
                        continue
                    num_posts = self._safe_int(row.get("num_posts"), 0)
                    num_comments = self._safe_int(row.get("num_comments"), 0)
                    total_votes = num_posts + num_comments
                    communities = row.get("communities_followed") or []
                    top_comms = {}
                    try:
                        for c in communities:
                            top_comms[str(c)] = 1
                    except Exception:
                        top_comms = {}
                    self.user_profiles[uid] = {
                        "total_votes": total_votes,
                        "top_communities": top_comms,
                        "num_posts": num_posts,
                        "num_comments": num_comments,
                    }
                logger.info(f"[LIGHT-H] built profiles for {len(self.user_profiles)} users")

            # load votes -> build user_post_matrix + similarity
            votes_df = self._load_votes_df()
            if votes_df is not None and not votes_df.empty:
                upr = votes_df[votes_df["target_type"] == "post"]
                if upr.empty:
                    logger.info("[LIGHT-H] no post votes found; collaborative disabled")
                    self.user_post_matrix = None
                    self.user_similarity_df = None
                else:
                    matrix = upr.pivot_table(index="user_id", columns="target_id", values="value", fill_value=0)
                    self.user_post_matrix = matrix
                    try:
                        sim = cosine_similarity(matrix)
                        self.user_similarity_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
                        logger.info(f"[LIGHT-H] built user-post matrix {matrix.shape}")
                    except Exception as e:
                        logger.exception("[LIGHT-H] similarity build failed: %s", e)
                        self.user_similarity_df = None
            else:
                logger.info("[LIGHT-H] votes collection empty; collaborative disabled")
                self.user_post_matrix = None
                self.user_similarity_df = None

            logger.info("[LIGHT-H] initialization complete")
        except Exception as e:
            logger.exception("[LIGHT-H] init failed: %s", e)
            # fallback to empty
            self.posts_df = pd.DataFrame()
            self.user_profiles = {}
            self.user_post_matrix = None
            self.user_similarity_df = None

    def _load_votes_df(self) -> pd.DataFrame:
        """
        Flatten votes collection into (user_id, target_id, value, target_type) rows.
        Expects documents shaped like your schema (votes.post.target_ids + value).
        """
        try:
            rows = []
            coll = self.db.get_collection("votes")
            for doc in coll.find({}):
                uid = doc.get("user_id")
                uid = str(uid) if uid is not None else str(doc.get("_id", ""))
                votes = doc.get("votes", {}) or {}
                post_votes = votes.get("post") or {}
                if isinstance(post_votes, dict):
                    target_ids = post_votes.get("target_ids", []) or []
                    value = post_votes.get("value", 0) or 0
                    try:
                        value = self._safe_int(value, 0)
                    except Exception:
                        value = 0
                    for tid in target_ids:
                        rows.append({
                            "user_id": uid,
                            "target_id": str(tid),
                            "value": value,
                            "target_type": "post"
                        })
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows)
        except Exception as e:
            logger.exception("[LIGHT-H] load_votes_df error: %s", e)
            return pd.DataFrame()

    # --------------------------- cold-start detection ---------------------------
    def is_cold_start_user(self, user_id: str) -> bool:
        uid = str(user_id)
        prof = self.user_profiles.get(uid)
        if prof is None:
            # user not present in profile store -> treat as cold-start
            return True
        return prof.get("total_votes", 0) == 0

    # --------------------------- scoring helpers ---------------------------
    def _get_cold_start_score(self, user_id: str, post_id: str) -> float:
        # community match (50%), popularity (30%), user activity (20%)
        uid = str(user_id)
        pid = str(post_id)
        user_prof = self.user_profiles.get(uid, {"total_votes": 0, "top_communities": {}, "num_posts": 0, "num_comments": 0})

        if self.posts_df is None or self.posts_df.empty:
            return 0.5

        post = self.posts_df[self.posts_df["post_id"] == pid]
        if post.shape[0] == 0:
            return 0.5

        post_community = str(post["community_id"].values[0]) if "community_id" in post.columns else ""
        post_score = float(post["score"].values[0]) if "score" in post.columns else 0.0

        community_match = 1.0 if post_community and post_community in user_prof.get("top_communities", {}) else 0.5
        popularity_score = min(1.0, post_score / 100.0)
        user_activity = min(1.0, (user_prof.get("num_posts", 0) + user_prof.get("num_comments", 0)) / 100.0)

        cold_score = (community_match * 0.5) + (popularity_score * 0.3) + (user_activity * 0.2)
        return float(np.clip(cold_score, 0.0, 1.0))

    def _get_collaborative_score(self, user_id: str, post_id: str) -> float:
        try:
            if self.user_similarity_df is None or self.user_post_matrix is None:
                return 0.5
            uid = str(user_id)
            pid = str(post_id)
            if uid not in self.user_similarity_df.index:
                return 0.5

            sims = self.user_similarity_df[uid].nlargest(6)
            sims = sims.iloc[1:6] if len(sims) > 1 else sims.iloc[0:0]

            votes = []
            for sim_uid in sims.index:
                if pid in self.user_post_matrix.columns and sim_uid in self.user_post_matrix.index:
                    vote = self.user_post_matrix.at[sim_uid, pid]
                    if vote != 0:
                        votes.append(vote)
            if votes:
                return float(np.clip(np.mean(votes), 0.0, 1.0))
            return 0.5
        except Exception as e:
            logger.exception("[LIGHT-H] collaborative score error: %s", e)
            return 0.5

    # --------------------------- recommendation generation ---------------------------
    def get_cold_start_recommendations(self, user_id: str, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k
        if self.posts_df is None or self.posts_df.empty:
            return []

        candidates = []
        for _, row in self.posts_df.iterrows():
            pid = str(row["post_id"])
            if "status" in row and row["status"] != "active":
                continue
            cold_score = self._get_cold_start_score(user_id, pid)
            collab_score = self._get_collaborative_score(user_id, pid)
            final_score = 0.6 * cold_score + 0.4 * collab_score
            candidates.append({"item_id": pid, "score": float(final_score)})

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top = candidates[:top_k]
        for i, item in enumerate(top, 1):
            item["rank"] = i

        # cache to upstash for faster future reads (best-effort)
        try:
            self.cache.store_user_recommendations(user_id, top)
        except Exception:
            logger.exception("[LIGHT-H] caching cold-start recs failed")

        return top

    # --------------------------- public API ---------------------------
    def get_hybrid_recommendations(self, user_id: str) -> Dict[str, Any]:
        uid = str(user_id)
        # 1) check cache
        try:
            cached = self.cache.get_user_recommendations(uid)
            if cached:
                return {"user_id": uid, "recommendations": cached, "source": "cache", "strategy": "cache"}
        except Exception:
            logger.exception("[LIGHT-H] cache read failed")

        # 2) cold-start => generate heuristics-based recs synchronously
        if self.is_cold_start_user(uid):
            logger.info("[LIGHT-H] cold-start detected for %s", uid)
            recs = self.get_cold_start_recommendations(uid, self.top_k)
            return {"user_id": uid, "recommendations": recs, "source": "cold_start", "strategy": "cold_start"}

        # 3) not cold-start & not cached -> we don't attempt heavy ML here; caller can trigger offline job
        logger.info("[LIGHT-H] no cache and not cold-start for %s", uid)
        return {"user_id": uid, "recommendations": None, "source": "none", "strategy": "none"}


# singleton accessor
_recommender: Optional[LightTopKRecommender] = None


def get_recommender() -> LightTopKRecommender:
    global _recommender
    if _recommender is None:
        _recommender = LightTopKRecommender()
    return _recommender
