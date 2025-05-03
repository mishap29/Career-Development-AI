import openai
import os
from typing import Dict, Optional, List

class PersonaModel:
    def __init__(self, personas: Dict[str, Dict], api_key: Optional[str] = None, system_prompt: Optional[str] = None):
        """
        Initialize with a dictionary of personas and their attributes.
        """
        self.personas = personas
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.system_prompt = system_prompt or (
            "You are a career advisor who specializes in tailoring advice to different personas (e.g., student, professional, career changer). "
            "You provide nuanced, empathetic, and actionable advice based on the user's persona and context."
        )
        self.user_history = []

    def add_to_history(self, persona_id: str, context: Dict, response: str):
        self.user_history.append({"persona": persona_id, "context": context, "advice": response})
        if len(self.user_history) > 5:
            self.user_history.pop(0)

    def generate_personalized_advice(self, persona_id: str, context: Dict, few_shot_examples: Optional[List] = None) -> str:
        persona = self.personas.get(persona_id, {})
        prompt = f"Give career advice for a user with persona: {persona_id} and context: {context}. Persona details: {persona}"
        messages = [{"role": "system", "content": self.system_prompt}]
        if few_shot_examples:
            for ex in few_shot_examples:
                messages.append({"role": "user", "content": ex[0]})
                messages.append({"role": "assistant", "content": ex[1]})
        for turn in self.user_history[-3:]:
            messages.append({"role": "user", "content": f"Persona: {turn['persona']}, Context: {turn['context']}"})
            messages.append({"role": "assistant", "content": turn["advice"]})
        messages.append({"role": "user", "content": prompt})
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        advice = response.choices[0].message.content
        self.add_to_history(persona_id, context, advice)
        return advice 