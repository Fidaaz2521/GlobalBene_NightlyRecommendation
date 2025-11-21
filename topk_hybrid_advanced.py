import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

from config import TOP_K
from model_loader import get_recommendation_model          # ✅ only PKL model from here
from embedding_generator import get_embedding_generator    # ✅ SBERT embedder from here
from faiss_indexer import get_faiss_indexer
from database import get_mongo_connection
from upstash_client import upstash_client  # your existing Upstash wrapper

logger = logging.getLogger(__name__)


class AdvancedTopKRecommender:
    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k

        # existing models / indexers
        try:
            self.pkl_model = get_recommendation_model()
        except Exception:
            self.pkl_model = None

        try:
            self.embedder = get_embedding_generator()
        except Exception:
            self.embedder = None

        try:
            self.indexer = get_faiss_indexer()
        except Exception:
            self.indexer = None

        # use your upstash client as cache manager
        self.cache = upstash_client

        # mongo connection helper (your MongoDBConnection instance)
        self.db_conn = get_mongo_connection()  # has .db attribute
        self.db = self.db_conn.db

        # heuristic / collaborative data
        self.posts_df: pd.DataFrame = pd.DataFrame()
        self.user_post_matrix: pd.DataFrame | None = None
        self.user_similarity_df: pd.DataFrame | None = None
        self.user_profiles: Dict[str, Dict[str, Any]] = {}

        # initialize heuristic data at startup
        self._init_heuristics_data()

    # ----------------- small helpers -----------------
    def _safe_int(self, value, default: int = 0) -> int:
        """
        Safely convert value to int:
        - Treats None/NaN as default
        - Handles strings, floats, etc.
        """
        import math

        if value is None:
            return default
        try:
            if isinstance(value, float) and math.isnan(value):
                return default
            return int(value)
        except Exception:
            return default

    # ----------------- Initialization helpers -----------------
    def _init_heuristics_data(self):
        logger.info("[HEURISTICS] initializing data...")
        try:
            # -------- posts --------
            posts_df = self.db_conn.get_posts()
            if posts_df is None or posts_df.empty:
                logger.warning("[HEURISTICS] no posts found")
                self.posts_df = pd.DataFrame()
            else:
                # rename _id -> post_id is already done in get_posts()
                # ensure post_id is string
                if "post_id" in posts_df.columns:
                    posts_df["post_id"] = posts_df["post_id"].astype(str)

                # ensure community_id is string if exists
                if "community_id" in posts_df.columns:
                    posts_df["community_id"] = posts_df["community_id"].astype(str)

                # ensure score column exists
                if "score" not in posts_df.columns:
                    posts_df["score"] = 0

                self.posts_df = posts_df
                logger.info(f"[HEURISTICS] loaded {len(self.posts_df)} posts")

            # -------- user profiles --------
            users_df = self.db_conn.get_users()
            if users_df is None or users_df.empty:
                logger.warning("[HEURISTICS] no users found")
                self.user_profiles = {}
            else:
                self.user_profiles = {}
                for _, row in users_df.iterrows():
                    uid = str(row.get("user_id") or row.get("_id"))
                    if not uid:
                        continue

                    num_posts = self._safe_int(row.get("num_posts"), 0)
                    num_comments = self._safe_int(row.get("num_comments"), 0)

                    total_votes = num_posts + num_comments

                    communities = row.get("communities_followed") or []
                    top_communities = {}
                    try:
                        for c in communities:
                            top_communities[str(c)] = 1
                    except Exception:
                        top_communities = {}

                    self.user_profiles[uid] = {
                        "total_votes": total_votes,
                        "top_communities": top_communities,
                        "num_posts": num_posts,
                        "num_comments": num_comments,
                    }

                logger.info(f"[HEURISTICS] built user_profiles for {len(self.user_profiles)} users")

            # -------- votes / collaborative matrix --------
            votes_df = self._load_votes_df()
            if votes_df is not None and not votes_df.empty:
                try:
                    upr = votes_df[votes_df["target_type"] == "post"]
                    if upr.empty:
                        logger.info("[HEURISTICS] no post votes in votes_df; collaborative disabled")
                        self.user_post_matrix = None
                        self.user_similarity_df = None
                    else:
                        matrix = upr.pivot_table(
                            index="user_id",
                            columns="target_id",
                            values="value",
                            fill_value=0,
                        )
                        self.user_post_matrix = matrix

                        sim = cosine_similarity(matrix)
                        self.user_similarity_df = pd.DataFrame(
                            sim,
                            index=matrix.index,
                            columns=matrix.index,
                        )
                        logger.info(f"[HEURISTICS] built user-post matrix {matrix.shape}")
                except Exception as e:
                    logger.exception(f"[HEURISTICS] error building matrix/similarity: {e}")
                    self.user_post_matrix = None
                    self.user_similarity_df = None
            else:
                logger.info("[HEURISTICS] no votes data found; collaborative disabled")
                self.user_post_matrix = None
                self.user_similarity_df = None

            logger.info("[HEURISTICS] init complete")
        except Exception as e:
            logger.exception(f"[HEURISTICS] init failed: {e}")
            # graceful fallback
            self.posts_df = pd.DataFrame()
            self.user_profiles = {}
            self.user_post_matrix = None
            self.user_similarity_df = None

    def _load_votes_df(self) -> pd.DataFrame:
        """
        Read votes collection and flatten to rows:
        Each votes document may look like your example:
        {
          _id: ...,
          user_id: ...,
          votes: {
            post: { target_ids: [...], value: 1 },
            comment: { ... }
          }
        }
        We'll expand post votes into rows:
        user_id, target_id, value, target_type='post'
        """
        try:
            rows = []
            coll = self.db.get_collection("votes")
            for doc in coll.find({}):
                # user id stored as user_id (ObjectId or string)
                uid = doc.get("user_id")
                uid = str(uid) if uid is not None else str(doc.get("_id", ""))

                votes = doc.get("votes", {}) or {}
                post_votes = votes.get("post") or {}
                if isinstance(post_votes, dict):
                    target_ids = post_votes.get("target_ids", []) or []
                    value = post_votes.get("value", 0) or 0
                    value = self._safe_int(value, 0)

                    for tid in target_ids:
                        rows.append(
                            {
                                "user_id": uid,
                                "target_id": str(tid),
                                "value": value,
                                "target_type": "post",
                            }
                        )

                # You can also flatten comment votes here if you need them

            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows)
        except Exception as e:
            logger.exception(f"[HEURISTICS] load_votes_df error: {e}")
            return pd.DataFrame()

    # ----------------- Cold-start detection -----------------
    def is_cold_start_user(self, user_id: str) -> bool:
        uid = str(user_id)
        prof = self.user_profiles.get(uid)
        if prof is None:
            return True
        return prof.get("total_votes", 0) == 0

    # ----------------- Scoring functions -----------------
    def _get_cold_start_score(self, user_id: str, post_id: str) -> float:
        """
        Heuristic: community match (50%), popularity (30%), user activity (20%)
        Returns value between 0 and 1
        """
        uid = str(user_id)
        pid = str(post_id)
        user_prof = self.user_profiles.get(
            uid,
            {
                "total_votes": 0,
                "top_communities": {},
                "num_posts": 0,
                "num_comments": 0,
            },
        )

        if self.posts_df is None or self.posts_df.empty:
            return 0.5

        post = self.posts_df[self.posts_df["post_id"] == pid]
        if post.shape[0] == 0:
            return 0.5

        post_community = str(post["community_id"].values[0]) if "community_id" in post.columns else ""
        post_score = float(post["score"].values[0]) if "score" in post.columns else 0.0

        community_match = 1.0 if post_community and post_community in user_prof.get("top_communities", {}) else 0.5
        popularity_score = min(1.0, post_score / 100.0)
        user_activity = min(
            1.0,
            (user_prof.get("num_posts", 0) + user_prof.get("num_comments", 0)) / 100.0,
        )

        cold_score = (community_match * 0.5) + (popularity_score * 0.3) + (user_activity * 0.2)
        return float(np.clip(cold_score, 0.0, 1.0))

    def _get_collaborative_score(self, user_id: str, post_id: str) -> float:
        """
        Look at top-K similar users and average their votes on post_id.
        Returns 0.5 (neutral) if no evidence.
        """
        try:
            if self.user_similarity_df is None or self.user_post_matrix is None:
                return 0.5

            uid = str(user_id)
            pid = str(post_id)

            if uid not in self.user_similarity_df.index:
                return 0.5

            sims = self.user_similarity_df[uid].nlargest(6)
            # skip self
            sims = sims.iloc[1:6] if len(sims) > 1 else sims.iloc[0:0]

            votes = []
            for sim_uid in sims.index:
                if pid in self.user_post_matrix.columns and sim_uid in self.user_post_matrix.index:
                    vote = self.user_post_matrix.at[sim_uid, pid]
                    if vote != 0:
                        votes.append(vote)

            if votes:
                score = float(np.clip(np.mean(votes), 0.0, 1.0))
                return score

            return 0.5
        except Exception as e:
            logger.exception(f"[HEURISTICS] collaborative score error: {e}")
            return 0.5

    # ----------------- Cold-start recommendation generator -----------------
    def get_cold_start_recommendations(self, user_id: str, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k
        if self.posts_df is None or self.posts_df.empty:
            return []

        candidates: List[Dict[str, Any]] = []
        for _, row in self.posts_df.iterrows():
            pid = str(row["post_id"])

            # only rank active posts if status exists
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

        # cache into upstash for quicker subsequent hits
        try:
            self.cache.store_user_recommendations(user_id, top)
        except Exception:
            logger.exception("[HEURISTICS] failed caching cold-start recs")

        return top

    # ----------------- Public entry for API -----------------
    def get_hybrid_recommendations(self, user_id: str) -> Dict[str, Any]:
        """
        Called by FastAPI:
        - If cache hit -> return cached
        - If cold-start user -> compute and return cold_start recs (and cache them)
        - Otherwise -> return {'recommendations': None, 'source':'on_demand'} so caller may enqueue Celery
        """
        uid = str(user_id)

        # cache check (Upstash)
        try:
            cached = self.cache.get_user_recommendations(uid)
            if cached:
                return {
                    "user_id": uid,
                    "recommendations": cached,
                    "source": "cache",
                    "strategy": "cache",
                }
        except Exception:
            logger.exception("Cache read failed")

        # cold start?
        if self.is_cold_start_user(uid):
            logger.info(f"[HYBRID] Cold-start user detected: {uid}")
            recs = self.get_cold_start_recommendations(uid, self.top_k)
            return {
                "user_id": uid,
                "recommendations": recs,
                "source": "cold_start",
                "strategy": "cold_start",
            }

        # not cold-start and not cached -> caller should enqueue celery
        logger.info(f"[HYBRID] Cache miss and not cold-start for {uid}")
        return {
            "user_id": uid,
            "recommendations": None,
            "source": "on_demand",
            "strategy": "enqueue",
        }


# singleton
_recommender = None


def get_recommender():
    global _recommender
    if _recommender is None:
        _recommender = AdvancedTopKRecommender()
    return _recommender
