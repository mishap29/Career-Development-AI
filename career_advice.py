from typing import Dict, List

class CareerAdvice:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def generate_advice(self, user_profile: Dict, personal_qualities: List[str], advice_db: List[Dict]) -> str:
        """
        Provide career advice based on user profile, personal qualities, and advice database using embeddings.
        """
        user_text = user_profile.get('text', '') + ' ' + ' '.join(personal_qualities)
        user_emb = self.embedding_model.encode(user_text)
        best_advice = None
        best_score = -1
        for advice in advice_db:
            advice_emb = self.embedding_model.encode(advice['text'])
            score = float(user_emb @ advice_emb) / (float((user_emb ** 2).sum()) ** 0.5 * float((advice_emb ** 2).sum()) ** 0.5)
            if score > best_score:
                best_score = score
                best_advice = advice['text']
        return best_advice or "No suitable advice found." 