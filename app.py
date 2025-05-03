import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import os
import random
from dotenv import load_dotenv
import openai
import numpy as np
import json
import pandas as pd

# Load Claude LLMs
with open("claude_llms.json") as f:
    claude_llms = json.load(f)["llms"]

# Import your models
from models.career_advisor_model import CareerAdvisorModel
from models.resume_generator import ResumeGenerator
from models.interview_preparation import InterviewPreparationChatbot
from models.persona_model import PersonaModel
from models.text_to_image import TextToImageModel
from models.fine_tuned_model import FineTunedCareerModel

# Import utils
from utils.data_preprocessing import DataPreprocessor
from utils.job_matching import JobMatcher
from utils.career_advice import CareerAdvice
from utils.embeddings import EmbeddingManager
from utils.user_interface import UserInterface

# OpenAI Embedding Model
class OpenAIEmbeddingModel:
    def __init__(self, model_name="text-embedding-ada-002"):
        self.model_name = model_name
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def encode(self, text):
        response = openai.Embedding.create(
            input=[text],
            model=self.model_name
        )
        emb = response['data'][0]['embedding']
        return np.array(emb)

# Initialize models and utilities
embedding_model = OpenAIEmbeddingModel()
data_preprocessor = DataPreprocessor()
job_matcher = JobMatcher(embedding_model)
user_interface = UserInterface()
career_advice = CareerAdvice(embedding_model)
embedding_manager = EmbeddingManager(embedding_model)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Model selection from claude_llms.json
model_names = [llm["name"] for llm in claude_llms]
selected_model_name = st.sidebar.selectbox("Choose a Claude LLM", model_names)
selected_model_id = next(llm["id"] for llm in claude_llms if llm["name"] == selected_model_name)

# Initialize models with selected model (where applicable)
career_advisor = CareerAdvisorModel(llms=[selected_model_id, "gpt-4", "claude-3-opus"])
resume_gen = ResumeGenerator(model_path="fine-tuned-resume-model")
interview_bot = InterviewPreparationChatbot(api_key=api_key)
persona_model = PersonaModel(personas={"student": {}, "professional": {}})
text2img = TextToImageModel(api_key=api_key)
fine_tuned = FineTunedCareerModel(model_path="fine-tuned-career-model")

st.title("Career Development AI")

# Define menu options
MENU_OPTIONS = [
    "Career Advice",
    "Resume Generation",
    "Interview Preparation",
    "Persona-based Advice",
    "Text-to-Image",
    "User Profile Upload",
    "Data Preprocessing",
    "Embeddings & RAG"
]

choice = st.sidebar.selectbox("Choose a feature", MENU_OPTIONS)

# Helper to get a random motivational poster
posters_dir = "assets/motivation_posters"
def get_random_poster():
    posters = [f for f in os.listdir(posters_dir) if f.endswith((".jpg", ".png"))]
    if posters:
        return os.path.join(posters_dir, random.choice(posters))
    return None

# Helper to list resume templates
templates_dir = "assets/resume_templates"
def get_resume_templates():
    # Only include JPEG images
    return [f for f in os.listdir(templates_dir) if f.lower().endswith(('.jpg', '.jpeg'))]

user_profile = st.session_state.get("user_profile", {})

if choice == "Career Advice":
    default_text = user_profile.get("summary", "") or user_profile.get("skills", "") or ""
    user_input = st.text_area("Describe your career question or situation:", value=default_text)
    qualities = st.text_input("List your personal qualities (comma separated):", value=user_profile.get("skills", ""))
    if st.button("Get Advice"):
        advice_db = [{"text": "Be proactive in networking."}, {"text": "Tailor your resume for each job."}]
        qualities_list = [q.strip() for q in qualities.split(",") if q.strip()]
        # Use uploaded profile if no new input
        profile_for_advice = user_profile.copy()
        if user_input:
            profile_for_advice["text"] = user_input
        st.write(career_advisor.generate_advice(profile_for_advice, qualities_list, advice_db))

elif choice == "Resume Generation":
    st.subheader("Generate Resume from Uploaded Profile")
    if user_profile:
        st.write("Using uploaded profile:", user_profile)
        if st.button("Generate Resume"):
            st.write(resume_gen.generate_resume(user_profile))
    else:
        st.info("Please upload your profile in the 'User Profile Upload' section.")
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

elif choice == "Interview Preparation":
    default_field = user_profile.get("desired_role", "") or user_profile.get("current_title", "")
    default_position = user_profile.get("current_title", "")
    field = st.text_input("Job field (e.g., Software Engineering):", value=default_field)
    position = st.text_input("Job position (e.g., Backend Developer):", value=default_position)
    user_feeling = st.text_input("How are you feeling about your interview?")
    if st.button("Get Interview Question"):
        st.write(interview_bot.ask_question(field, position))
        if any(word in user_feeling.lower() for word in ["nervous", "anxious", "worried", "scared"]):
            poster_path = get_random_poster()
            if poster_path:
                st.image(poster_path, caption="You are stronger than you think! Stay motivated!")

elif choice == "Persona-based Advice":
    default_persona = user_profile.get("persona", "")
    default_context = user_profile.get("summary", "") or user_profile.get("skills", "") or ""
    persona_id = st.text_input("Persona (e.g., student, professional):", value=default_persona)
    context = st.text_area("Describe your situation:", value=default_context)
    if st.button("Get Personalized Advice"):
        context_dict = {"context": context}
        st.write(persona_model.generate_personalized_advice(persona_id, context_dict))

elif choice == "Text-to-Image":
    default_prompt = f"Motivational poster for a {user_profile.get('desired_role', 'professional')}" if user_profile else ""
    prompt = st.text_area("Describe the visual content you want to generate:", value=default_prompt)
    if st.button("Generate Image"):
        if text2img:
            image_path = text2img.generate_image(prompt)
            st.image(image_path, caption="Generated Image")
        else:
            st.warning("Text-to-Image feature is not available.")

elif choice == "User Profile Upload":
    st.subheader("Upload Your LinkedIn Export or Resume")
    uploaded_file = st.file_uploader("Choose a CSV, PDF, or DOCX file", type=["csv", "pdf", "docx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            user_df = pd.read_csv(uploaded_file)
            st.write("Parsed User Profile Data:", user_df)
            # Store the first row as the active user profile (customize as needed)
            st.session_state["user_profile"] = user_df.iloc[0].to_dict()
        elif uploaded_file.name.endswith(".pdf") or uploaded_file.name.endswith(".docx"):
            st.info("Resume parsing for PDF/DOCX coming soon! For now, please upload a CSV export from LinkedIn.")

elif choice == "Data Preprocessing":
    st.subheader("Upload a CSV to Clean and Prepare")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Original Data", df)
        cleaned_df = data_preprocessor.clean_dataframe(df)
        st.write("Cleaned Data", cleaned_df)
        featured_df = data_preprocessor.extract_features(cleaned_df)
        st.write("With Features", featured_df)
    elif user_profile:
        st.write("Using uploaded user profile for preprocessing:")
        df = pd.DataFrame([user_profile])
        cleaned_df = data_preprocessor.clean_dataframe(df)
        st.write("Cleaned Data", cleaned_df)
        featured_df = data_preprocessor.extract_features(cleaned_df)
        st.write("With Features", featured_df)

elif choice == "Embeddings & RAG":
    st.subheader("Add Text to Embedding Store")
    new_text = st.text_input("Text to add:")
    if st.button("Add Embedding"):
        embedding_manager.add_embedding(new_text)
        st.success("Added!")
    query = st.text_input("Query for RAG:")
    persona_for_rag = user_profile.get("persona", "") if user_profile else ""
    if st.button("Retrieve with RAG"):
        from utils.embeddings import personalized_rag_prompt
        user_profile_rag = {"persona": persona_for_rag} if persona_for_rag else None
        rag_mode = st.selectbox("RAG Mode", ["default", "inspirational", "step-by-step"])
        st.write(embedding_manager.rag_generate(query, personalized_rag_prompt, user_profile_rag, rag_mode)) 