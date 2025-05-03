import os
import random
from dotenv import load_dotenv
import openai
import streamlit as st

load_dotenv()

class CareerChatbot:
    def __init__(self, api_key: str = None, posters_dir: str = "assets/motivation_posters"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.system_prompt = (
            "You are a kind, calming, and understanding career coach and interview preparation assistant. "
            "You help users explore how their skills can be applied to different careers or fields, "
            "and you provide thoughtful, encouraging, and practical advice."
        )
        self.posters_dir = posters_dir

    def chat(self, user_input: str, few_shot_examples=None) -> dict:
        """
        Respond to user input using OpenAI, with optional few-shot prompting.
        Returns a dict with 'response' and optional 'poster_path'.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        if few_shot_examples:
            for ex in few_shot_examples:
                messages.append({"role": "user", "content": ex[0]})
                messages.append({"role": "assistant", "content": ex[1]})
        messages.append({"role": "user", "content": user_input})
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        reply = response.choices[0].message.content
        # Check for nervousness/anxiety in user input
        if any(word in user_input.lower() for word in ["nervous", "anxious", "worried", "scared"]):
            poster_path = self.get_random_poster()
            return {"response": reply, "poster_path": poster_path}
        return {"response": reply}

    def get_random_poster(self):
        posters = [f for f in os.listdir(self.posters_dir) if f.endswith((".jpg", ".png"))]
        if posters:
            return os.path.join(self.posters_dir, random.choice(posters))
        return None

    def suggest_careers_for_skills(self, skills: list) -> str:
        """
        Suggest possible careers or fields based on a list of user skills.
        """
        prompt = (
            f"The user has the following skills: {', '.join(skills)}. "
            "What are some career paths or fields where these skills would be valuable? "
            "Please explain in a supportive and encouraging tone."
        )
        return self.chat(prompt)["response"]

# Example usage (for testing or integration):
if __name__ == "__main__":
    bot = CareerChatbot()
    print("Welcome to the Career Coaching Chatbot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        result = bot.chat(user_input)
        print("Bot:", result["response"])
        if "poster_path" in result:
            print(f"[Motivational poster: {result['poster_path']}]")

templates_dir = "assets/resume_templates"

def get_resume_templates():
    # Only include JPEG images
    return [f for f in os.listdir(templates_dir) if f.lower().endswith(('.jpg', '.jpeg'))]

# ... inside your Resume Generation section:
st.subheader("Resume Template Gallery")
for template in get_resume_templates():
    template_path = os.path.join(templates_dir, template)
    st.image(template_path, caption=template)
    with open(template_path, "rb") as file:
        st.download_button(
            label=f"Download {template}",
            data=file,
            file_name=template
        ) 