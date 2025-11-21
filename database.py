from pymongo import MongoClient
from config import MONGO_URI, DATABASE_NAME
import pandas as pd
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)

class MongoDBConnection:
    def __init__(self):
        try:
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[DATABASE_NAME]
            logger.info(f"✓ Connected to MongoDB: {DATABASE_NAME}")
        except Exception as e:
            logger.error(f"✗ MongoDB connection failed: {e}")
            raise
    
    def _convert_objectid_to_str(self, data):
        if isinstance(data, list):
            for item in data:
                if '_id' in item and isinstance(item['_id'], ObjectId):
                    item['_id'] = str(item['_id'])
        return data
    
    def get_users(self, filter_query=None):
        try:
            if filter_query is None:
                filter_query = {}
            
            users_bson = list(self.db['users'].find(filter_query))
            
            if not users_bson:
                logger.warning("No users found")
                return pd.DataFrame()
            
            users_bson = self._convert_objectid_to_str(users_bson)
            users_df = pd.DataFrame(users_bson)
            
            if '_id' in users_df.columns:
                users_df = users_df.rename(columns={'_id': 'user_id'})
            
            logger.info(f"✓ Fetched {len(users_df)} users")
            return users_df
            
        except Exception as e:
            logger.error(f"✗ Error fetching users: {e}")
            return pd.DataFrame()
    
    def get_posts(self, filter_query=None):
        try:
            if filter_query is None:
                filter_query = {}
            
            posts_bson = list(self.db['posts'].find(filter_query))
            
            if not posts_bson:
                logger.warning("No posts found")
                return pd.DataFrame()
            
            posts_bson = self._convert_objectid_to_str(posts_bson)
            posts_df = pd.DataFrame(posts_bson)
            
            if '_id' in posts_df.columns:
                posts_df = posts_df.rename(columns={'_id': 'post_id'})
            
            logger.info(f"✓ Fetched {len(posts_df)} posts")
            return posts_df
            
        except Exception as e:
            logger.error(f"✗ Error fetching posts: {e}")
            return pd.DataFrame()

_mongo_connection = None

def get_mongo_connection():
    global _mongo_connection
    if _mongo_connection is None:
        _mongo_connection = MongoDBConnection()
    return _mongo_connection


def get_all_users():
    """
    Return a list of all user dicts (not a DataFrame),
    so tasks.py can iterate over them.
    """
    users_df = get_mongo_connection().get_users()
    # Convert DataFrame to list of dicts
    return users_df.to_dict(orient='records') if not users_df.empty else []

def get_user_data(user_id):
    """
    Return a dict for the user with given user_id; None if not found.
    """
    users_df = get_mongo_connection().get_users({'_id': ObjectId(user_id)})
    if not users_df.empty:
        return users_df.iloc[0].to_dict()
    return None
