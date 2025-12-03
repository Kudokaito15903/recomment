import lightgbm as lgb
import numpy as np
import pickle
from loguru import logger

class Ranker:
    def __init__(self, model_path="models/ranker.pkl"):
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info("Loaded ranking model from {}".format(model_path))
        except Exception:
            self.model = None
            logger.warning("No ranking model found, will fallback")

    def rank(self, user_features, candidate_features):
        # user_features: np.array(1, f)
        # candidate_features: np.array(n, f)
        if candidate_features.size == 0:
            return np.array([])

        user_features = np.atleast_2d(user_features)
        n = candidate_features.shape[0]
        user_block = np.repeat(user_features, n, axis=0)
        X = np.hstack([user_block, candidate_features])
        if self.model:
            scores = self.model.predict(X)
        else:
            scores = np.random.rand(n)
        return scores
