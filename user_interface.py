from utils.data_preprocessing import DataPreprocessor
from utils.job_matching import JobMatcher
from utils.career_advice import CareerAdvice
from utils.embeddings import EmbeddingManager

class UserInterface:
    def __init__(self):
        self.app_purpose = (
            "This application is designed to help users with career development, including career advice, resume generation, interview preparation, job matching, and motivational support. "
            "If you have questions outside of these topics, please refer to a more general resource."
        )

    def get_input(self, prompt: str) -> str:
        return input(prompt)

    def display_output(self, message: str):
        print(message)

    def clarify_scope(self, user_input: str) -> str:
        # If user input is outside the app's scope, clarify the purpose
        keywords = ["weather", "sports", "news", "movie", "music"]
        if any(word in user_input.lower() for word in keywords):
            return self.app_purpose
        return "" 