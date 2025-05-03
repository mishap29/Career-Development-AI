import numpy as np
from typing import List, Dict

class JobMatcher:
    def __init__(self, embedding_model):
        """
        Initialize with an embedding model (should have an encode method).
        """
        self.embedding_model = embedding_model

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        """
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def match(self, user_profile: Dict, job_listings: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Match a user profile with job listings using embeddings. Returns top_k matches.
        """
        user_emb = self.embedding_model.encode(user_profile['text'])
        job_embs = [self.embedding_model.encode(job['text']) for job in job_listings]
        similarities = [self.compute_similarity(user_emb, job_emb) for job_emb in job_embs]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [job_listings[i] for i in top_indices] 