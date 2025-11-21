from sentence_transformers import SentenceTransformer
from config import SBERT_MODEL_NAME
import numpy as np

class EmbeddingGenerator:
    def __init__(self):
        # Download and cache SBERT model
        self.model = SentenceTransformer(SBERT_MODEL_NAME)
    
    def generate_post_embeddings(self, posts_df):
        """Convert posts to embeddings"""
        
        # Extract text from each post
        texts = []
        for idx, post in posts_df.iterrows():
            # Combine caption + body + title
            caption = str(post.get('caption', ''))
            body = str(post.get('body', ''))
            title = str(post.get('title', ''))
            
            combined_text = f"{caption} {body} {title}"
            texts.append(combined_text)
        
        # Convert all texts to embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Get post IDs
        post_ids = posts_df['post_id'].tolist()
        
        return embeddings, post_ids
    
    def generate_user_embeddings(self, users_df):
        """Convert user profiles to embeddings"""
        
        # Extract user profile text
        texts = []
        for idx, user in users_df.iterrows():
            username = str(user.get('username', ''))
            interests = str(user.get('interests', ''))
            bio = str(user.get('bio', ''))
            
            combined_text = f"{username} {bio} {interests}"
            texts.append(combined_text)
        
        # Convert to embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Get user IDs
        user_ids = users_df['user_id'].tolist()
        
        return embeddings, user_ids

_generator = None

def get_embedding_generator():
    global _generator
    if _generator is None:
        _generator = EmbeddingGenerator()
    return _generator
