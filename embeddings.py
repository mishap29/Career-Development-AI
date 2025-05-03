from typing import List, Dict, Callable, Optional
import numpy as np
import openai

class EmbeddingManager:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        # Store context as dicts with metadata
        self.embeddings = []  # List of dicts: {"text": ..., "embedding": ..., "meta": {...}}

    def add_embedding(self, text: str, meta: Optional[Dict] = None):
        emb = self.embedding_model.encode(text)
        self.embeddings.append({
            "text": text,
            "embedding": emb,
            "meta": meta or {}
        })

    def search(self, query: str, top_k: int = 3, filters: Optional[Dict] = None) -> List[Dict]:
        query_emb = self.embedding_model.encode(query)
        # Optionally filter by metadata
        candidates = self.embeddings
        if filters:
            candidates = [
                e for e in self.embeddings
                if all(e["meta"].get(k) == v for k, v in filters.items())
            ]
        scored = [
            (e, float(query_emb @ e["embedding"]) / (float((query_emb ** 2).sum()) ** 0.5 * float((e["embedding"] ** 2).sum()) ** 0.5))
            for e in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    def rag_generate(
        self,
        query: str,
        generator_fn: Callable[[str, List[Dict], Optional[Dict], str], str],
        user_profile: Optional[Dict] = None,
        rag_mode: str = "default",
        top_k: int = 3,
        filters: Optional[Dict] = None
    ) -> str:
        """
        Retrieve top_k relevant contexts (optionally filtered), and use them as context for a generator function (e.g., LLM).
        Pass user_profile and rag_mode for more personalized generation.
        """
        context_chunks = self.search(query, top_k=top_k, filters=filters)
        return generator_fn(query, context_chunks, user_profile, rag_mode)

# Example personalized RAG prompt generator for OpenAI

def personalized_rag_prompt(query, context_chunks, user_profile, rag_mode):
    context_texts = "\n\n".join([f"- {c['text']}" for c in context_chunks])
    persona = user_profile.get("persona", "user") if user_profile else "user"
    history = user_profile.get("history", "") if user_profile else ""
    mode_instruction = {
        "default": "",
        "inspirational": "Provide advice in an inspirational and motivational tone.",
        "step-by-step": "Break down your advice into clear, actionable steps."
    }.get(rag_mode, "")

    prompt = (
        f"You are a career development assistant helping a {persona}.\n"
        f"User history: {history}\n"
        f"Relevant information:\n{context_texts}\n\n"
        f"{mode_instruction}\n"
        f"User query: {query}\n"
        f"Please provide a detailed, personalized response."
    )
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a career development expert."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content 