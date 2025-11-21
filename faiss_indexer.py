# faiss_indexer.py

import faiss
import numpy as np
import pickle
import logging
from config import FAISS_INDEX_PATH

logger = logging.getLogger(__name__)

class FAISSIndexer:
    def __init__(self):
        self.index = None
        self.post_ids = []

    def create_index_cosine(self, embeddings, post_ids):
        """Build FAISS index using cosine similarity"""

        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)

        dimension = int(embeddings.shape[1])
        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)
        self.post_ids = post_ids

    def search(self, query_vector, k=50):
        """Search for K nearest neighbors"""

        if self.index is None:
            raise RuntimeError("FAISS index is not loaded. Call load_index() first.")

        query_vector = query_vector.astype('float32')
        query_vector = np.expand_dims(query_vector, axis=0)  # Make 2D

        faiss.normalize_L2(query_vector)

        distances, indices = self.index.search(query_vector, k)

        # indices is shape (1, k). Flatten it
        indices = indices[0]
        result_post_ids = [self.post_ids[idx] for idx in indices]

        return distances, result_post_ids

    def save_index(self, filepath=FAISS_INDEX_PATH):
        """Save index to disk"""

        faiss.write_index(self.index, filepath)

        ids_filepath = filepath.replace('.bin', '_ids.pkl')
        with open(ids_filepath, 'wb') as f:
            pickle.dump(self.post_ids, f)

    def load_index(self, filepath=FAISS_INDEX_PATH):
        """Load index from disk"""

        logger.info(f"🔍 Loading FAISS index from {filepath}")
        self.index = faiss.read_index(filepath)

        ids_filepath = filepath.replace('.bin', '_ids.pkl')
        with open(ids_filepath, 'rb') as f:
            self.post_ids = pickle.load(f)

        logger.info(f"✅ FAISS index loaded. #posts = {len(self.post_ids)}")


_indexer = None

def get_faiss_indexer():
    """
    Singleton accessor – always returns an indexer with index loaded.
    """
    global _indexer
    if _indexer is None:
        _indexer = FAISSIndexer()
        # Load index immediately when first accessed
        _indexer.load_index()
    return _indexer
