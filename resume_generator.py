import openai
import os
from typing import Dict, Optional, List

class ResumeGenerator:
    def __init__(self, model_path: str, api_key: Optional[str] = None, system_prompt: Optional[str] = None):
        """
        Initialize with the path or identifier for the fine-tuned resume generation model.
        """
        self.model_path = model_path
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.system_prompt = system_prompt or (
            "You are a professional resume writer. You create clear, concise, and impactful resumes tailored to the user's background and the job they are seeking."
        )
        self.user_history = []

    def add_to_history(self, user_data: Dict, response: str):
        self.user_history.append({"user": user_data, "resume": response})
        if len(self.user_history) > 5:
            self.user_history.pop(0)

    def generate_resume(self, user_data: Dict, job_title: Optional[str] = None, custom_instructions: Optional[str] = None, few_shot_examples: Optional[List] = None) -> str:
        """
        Generate a resume using the fine-tuned model and user data.
        """
        prompt = f"Generate a professional resume for the following information: {user_data}"
        if job_title:
            prompt += f"\nTarget job title: {job_title}"
        if custom_instructions:
            prompt += f"\nSpecial instructions: {custom_instructions}"
        messages = [{"role": "system", "content": self.system_prompt}]
        if few_shot_examples:
            for ex in few_shot_examples:
                messages.append({"role": "user", "content": ex[0]})
                messages.append({"role": "assistant", "content": ex[1]})
        for turn in self.user_history[-3:]:
            messages.append({"role": "user", "content": str(turn["user"])} )
            messages.append({"role": "assistant", "content": turn["resume"]})
        messages.append({"role": "user", "content": prompt})
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        resume = response.choices[0].message.content
        self.add_to_history(user_data, resume)
        return resume 