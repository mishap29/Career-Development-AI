import openai
import os
from typing import List, Dict, Optional

class CareerAdvisorModel:
    def __init__(self, llms: List[str], api_key: Optional[str] = None, system_prompt: Optional[str] = None):
        """
        Initialize with a list of LLM identifiers (e.g., model names or API endpoints).
        """
        self.llms = llms
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.system_prompt = system_prompt or (
            "You are a highly experienced, empathetic, and insightful career advisor. "
            "You provide actionable, personalized, and strategic career advice based on user input, context, and history."
        )
        self.user_history = []

    def select_llm(self, context: Dict) -> str:
        """
        Select an LLM based on the context (e.g., user preferences, topic).
        """
        # Example: select LLM based on context, fallback to first
        if context and 'preferred_llm' in context:
            return context['preferred_llm']
        return self.llms[0] if self.llms else "gpt-4"

    def add_to_history(self, user_input: str, response: str):
        self.user_history.append({"user": user_input, "advisor": response})
        if len(self.user_history) > 10:
            self.user_history.pop(0)

    def generate_advice(self, user_input: str, context: Dict = None, few_shot_examples: List = None) -> str:
        """
        Generate career advice using the selected LLM.
        """
        llm = self.select_llm(context or {})
        messages = [{"role": "system", "content": self.system_prompt}]
        # Add few-shot examples if provided
        if few_shot_examples:
            print("DEBUG: few_shot_examples =", few_shot_examples)
            for ex in few_shot_examples:
                if isinstance(ex, dict) and "text" in ex:
                    messages.append({"role": "assistant", "content": ex["text"]})
                elif isinstance(ex, (list, tuple)) and len(ex) == 2:
                    messages.append({"role": "user", "content": ex[0]})
                    messages.append({"role": "assistant", "content": ex[1]})
                else:
                    print("WARNING: Skipping invalid few_shot_example:", ex)
        # Add user history for context
        for turn in self.user_history[-5:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["advisor"]})
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Use the correct API method for the chat completion
        try:
            response = openai.chat.completions.create(
                model=llm,
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )
            advisor_reply = response.choices[0].message.content
            self.add_to_history(user_input, advisor_reply)
            return advisor_reply
        except Exception as e:
            return f"Error: {str(e)}\nYour API key may not have access to the selected model. Please check your OpenAI account and key permissions."
