import os
from typing import List, Dict
from dotenv import load_dotenv
import openai

load_dotenv()

class InterviewPreparationChatbot:
    def __init__(self, field_questions: Dict[str, List[str]] = None, api_key: str = None):
        """
        Initialize with a mapping from job fields to lists of interview questions and OpenAI API key.
        """
        self.field_questions = field_questions or {}
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key

    def get_questions_for_field(self, field: str, position: str = "") -> List[str]:
        """
        Use OpenAI to generate interview questions for the specified field and position.
        """
        prompt = (
            f"You are a recruiter for a {field} position. "
            f"Generate 5 interview questions that would be relevant for a candidate applying for the position of {position}."
        )
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert recruiter."},
                {"role": "user", "content": prompt}
            ]
        )
        questions = response.choices[0].message.content.strip().split('\n')
        # Clean up and return only non-empty questions
        return [q for q in questions if q.strip()]

    def ask_question(self, field: str, position: str = "") -> str:
        """
        Ask the first generated question for the given field and position using OpenAI.
        """
        questions = self.get_questions_for_field(field, position)
        return questions[0] if questions else "No questions available." 