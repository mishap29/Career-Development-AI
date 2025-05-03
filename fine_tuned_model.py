import openai
import os
from typing import Dict, List, Optional

class FineTunedCareerModel:
    def __init__(self, model_path: str, api_key: Optional[str] = None, system_prompt: Optional[str] = None):
        """
        Initialize with the path or identifier for the fine-tuned LLM.
        """
        self.model_path = model_path
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.system_prompt = system_prompt or (
            "You are a career development expert specializing in job matching and resume review. "
            "You provide detailed, actionable, and personalized feedback."
        )
        self.user_history = []

    def add_to_history(self, user_profile: Dict, job_listings: List[Dict], response: str):
        self.user_history.append({"profile": user_profile, "jobs": job_listings, "response": response})
        if len(self.user_history) > 5:
            self.user_history.pop(0)

    def job_matching(self, user_profile: Dict, job_listings: List[Dict], few_shot_examples: Optional[List] = None) -> str:
        """
        Match user profile to job listings using the fine-tuned model.
        """
        prompt = f"Match the following user profile to these job listings and explain the best matches:\nUser: {user_profile}\nJobs: {job_listings}"
        messages = [{"role": "system", "content": self.system_prompt}]
        if few_shot_examples:
            for ex in few_shot_examples:
                messages.append({"role": "user", "content": ex[0]})
                messages.append({"role": "assistant", "content": ex[1]})
        for turn in self.user_history[-3:]:
            messages.append({"role": "user", "content": f"User: {turn['profile']}, Jobs: {turn['jobs']}"})
            messages.append({"role": "assistant", "content": turn["response"]})
        messages.append({"role": "user", "content": prompt})
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        result = response.choices[0].message.content
        self.add_to_history(user_profile, job_listings, result)
        return result

    def resume_tips(self, resume_text: str, custom_instructions: Optional[str] = None, few_shot_examples: Optional[List] = None) -> str:
        """
        Provide resume tips using the fine-tuned model.
        """
        prompt = f"Review this resume and provide detailed, actionable tips for improvement:\n{resume_text}"
        if custom_instructions:
            prompt += f"\nSpecial instructions: {custom_instructions}"
        messages = [{"role": "system", "content": self.system_prompt}]
        if few_shot_examples:
            for ex in few_shot_examples:
                messages.append({"role": "user", "content": ex[0]})
                messages.append({"role": "assistant", "content": ex[1]})
        for turn in self.user_history[-3:]:
            messages.append({"role": "user", "content": f"Resume: {turn['profile']}"})
            messages.append({"role": "assistant", "content": turn["response"]})
        messages.append({"role": "user", "content": prompt})
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        tips = response.choices[0].message.content
        self.add_to_history({"resume": resume_text}, [], tips)
        return tips 