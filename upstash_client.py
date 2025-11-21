import json
import logging
from typing import Any, List, Dict, Optional
from upstash_redis import Redis
from upstash_redis.asyncio import Redis as AsyncRedis
import os
from dotenv import load_dotenv  


logger = logging.getLogger(__name__)

# Load .env as soon as this module is imported
load_dotenv()  



class UpstashClient:
    def __init__(self):
        """Initialize Upstash Redis client"""
        self.sync_redis = Redis(
            url=os.getenv("UPSTASH_REDIS_REST_URL"),
            token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
           
        )
        self.async_redis = AsyncRedis(
            url=os.getenv("UPSTASH_REDIS_REST_URL"),
            token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
        )

    def store_user_recommendations(
        self, 
        user_id: str, 
        recommendations: List[Dict[str, Any]], 
        expiry_hours: int = 24
    ) -> bool:
        """
        Store top-K recommendations for a user in Upstash Redis
        
        Args:
            user_id: Unique user identifier
            recommendations: List of recommendation dicts with item_id and score
            expiry_hours: TTL in hours (default: 24)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"recommendations:{user_id}"
            
            # Serialize recommendations to JSON
            value = json.dumps({
                "user_id": user_id,
                "recommendations": recommendations,
                "timestamp": int(__import__('time').time())
            })
            
            # Store with expiration
            expiry_seconds = expiry_hours * 3600
            self.sync_redis.setex(key, expiry_seconds, value)
            
            logger.info(f"✓ Stored {len(recommendations)} recommendations for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error storing recommendations for {user_id}: {str(e)}")
            return False

    def get_user_recommendations(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve cached recommendations for a user (< 0.1s response)
        
        Args:
            user_id: Unique user identifier
        
        Returns:
            List of recommendations or None if not cached
        """
        try:
            key = f"recommendations:{user_id}"
            value = self.sync_redis.get(key)
            
            if value:
                data = json.loads(value)
                logger.info(f"✓ Cache hit for user {user_id}")
                return data["recommendations"]
            
            logger.info(f"✗ Cache miss for user {user_id}")
            return None
            
        except Exception as e:
            logger.error(f"✗ Error retrieving recommendations for {user_id}: {str(e)}")
            return None

    def store_batch_recommendations(
        self, 
        user_recommendations: Dict[str, List[Dict[str, Any]]], 
        expiry_hours: int = 24
    ) -> Dict[str, bool]:
        """
        Store recommendations for multiple users efficiently using pipeline
        
        Args:
            user_recommendations: Dict mapping user_id to their recommendations
            expiry_hours: TTL in hours
        
        Returns:
            Dict with success status for each user
        """
        try:
            pipeline = self.sync_redis.pipeline()
            expiry_seconds = expiry_hours * 3600
            
            results = {}
            for user_id, recommendations in user_recommendations.items():
                key = f"recommendations:{user_id}"
                value = json.dumps({
                    "user_id": user_id,
                    "recommendations": recommendations,
                    "timestamp": int(__import__('time').time())
                })
                
                pipeline.setex(key, expiry_seconds, value)
                results[user_id] = True
            
            # Execute pipeline
            pipeline.exec()
            logger.info(f"✓ Batch stored recommendations for {len(user_recommendations)} users")
            return results
            
        except Exception as e:
            logger.error(f"✗ Error in batch storage: {str(e)}")
            return {uid: False for uid in user_recommendations.keys()}

    def clear_recommendations(self, user_id: str) -> bool:
        """Delete recommendations for a user"""
        try:
            key = f"recommendations:{user_id}"
            self.sync_redis.delete(key)
            logger.info(f"✓ Cleared recommendations for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"✗ Error clearing recommendations: {str(e)}")
            return False

# Initialize singleton instance
upstash_client = UpstashClient()
