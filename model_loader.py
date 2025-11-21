# # import logging

# # logger = logging.getLogger(__name__)

# # class DummyModel:
# #     """Dummy model - PKL corrupted, using SBERT+FAISS only"""
    
# #     def __init__(self):
# #         self.model_type = "dummy"
# #         logger.info("⚠️ PKL corrupted - Using SBERT+FAISS only (no PKL blend)")
    
# #     def get_recommendations(self, user_id, top_k=50):
# #         """Return empty list - will use SBERT+FAISS only"""
# #         return []

# # _model = None

# # def get_recommendation_model():
# #     global _model
# #     if _model is None:
# #         _model = DummyModel()
# #     return _model


# import pickle
# import logging
# from config import MODEL_PATH

# logger = logging.getLogger(__name__)

# class RecommendationModel:
#     def __init__(self, model_path=MODEL_PATH):
#         self.model = None
#         self.model_path = model_path
#         self.model_type = None
#         self.load_model()
    
#     def load_model(self):
#         try:
#             with open(self.model_path, 'rb') as file:
#                 self.model = pickle.load(file)
            
#             logger.info(f"✓ Loaded model from: {self.model_path}")
#             self._detect_model_type()
            
#         except FileNotFoundError:
#             logger.error(f"✗ Model file not found: {self.model_path}")
#             raise
#         except Exception as e:
#             logger.error(f"✗ Error loading model: {e}")
#             raise
    
#     def _detect_model_type(self):
#         if isinstance(self.model, dict):
#             self.model_type = "dictionary"
#             logger.info("  → Type: Dictionary {user_id: [post_ids]}")
#         elif hasattr(self.model, 'predict'):
#             self.model_type = "sklearn"
#             logger.info("  → Type: Scikit-learn model")
#         elif hasattr(self.model, 'recommend'):
#             self.model_type = "custom_recommend"
#             logger.info("  → Type: Custom with recommend() method")
#         else:
#             self.model_type = "unknown"
#             logger.warning("  → Type: Unknown")
    
#     def get_recommendations(self, user_id, top_k=50):
#         try:
#             if self.model_type == "dictionary":
#                 recs = self.model.get(str(user_id), [])
#                 return list(recs)[:top_k]
            
#             elif self.model_type == "sklearn":
#                 return list(self.model.predict(user_id, top_k))
            
#             elif self.model_type == "custom_recommend":
#                 return list(self.model.recommend(user_id, top_k))
            
#             else:
#                 logger.warning(f"Unable to get recommendations for {user_id}")
#                 return []
        
#         except Exception as e:
#             logger.error(f"Error getting recommendations: {e}")
#             return []

# _model = None

# def get_recommendation_model():
#     global _model
#     if _model is None:
#         _model = RecommendationModel()
#     return _model
import logging

logger = logging.getLogger(__name__)

class DummyModel:
    """Dummy model - PKL corrupted, using SBERT+FAISS only"""
    
    def __init__(self):
        self.model_type = "dummy"
        logger.info("⚠️ PKL corrupted or disabled - Using SBERT+FAISS only (no PKL blend)")
    
    def get_recommendations(self, user_id, top_k=50):
        """Return empty list - SBERT+FAISS will handle recommendations."""
        return []

_model = None

def get_recommendation_model():
    global _model
    if _model is None:
        _model = DummyModel()
    return _model

