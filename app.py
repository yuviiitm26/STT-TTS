import streamlit as st
import speech_recognition as sr
import pyttsx3
import tempfile
import os
from config import LLM
from ingestion import build_vector_db, build_knowledge_graph
from langchain_core.prompts import PromptTemplate

# --- Page Configuration ---
st.set_page_config(page_title="System Architecture AI", page_icon="🤖", layout="centered")

st.title("🤖 System Architecture Assistant")
st.markdown("Ask questions about your infrastructure using text or your voice!")

# --- Initialize Session State ---
if "vector_db" not in st.session_state:
    with st.spinner("Building Knowledge Bases (FAISS & Graph)..."):
        st.session_state.vector_db = build_vector_db()
        st.session_state.graph = build_knowledge_graph()
        st.success("Knowledge Bases Loaded!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- TTS Helper Function ---
def generate_audio_file(text):
    """Converts text to an offline audio file so the browser can play it."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 175)
    
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, "output.wav")
    
    engine.save_to_file(text, audio_path)
    engine.runAndWait()
    return audio_path

# --- Display Chat History ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input Handlers (Voice & Text) ---
# 1. Voice Input Widget
recorded_audio = st.audio_input("Speak your question:")
# 2. Standard Text Input
text_input = st.chat_input("Or type your question here...")

# Determine which input the user used
user_question = None

if recorded_audio:
    with st.spinner("Transcribing your voice..."):
        recognizer = sr.Recognizer()
        # Read the audio bytes directly from the Streamlit widget
        with sr.AudioFile(recorded_audio) as source:
            audio_data = recognizer.record(source)
            try:
                user_question = recognizer.recognize_google(audio_data)
            except:
                st.error("Could not understand the audio. Please try again.")

elif text_input:
    user_question = text_input

# --- Process the Question ---
if user_question:
    # Display User Message
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # Retrieve Context & Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Retrieve Context
            vector_results = st.session_state.vector_db.similarity_search(user_question, k=2)
            vector_context = "\n".join([res.page_content for res in vector_results])
            graph_context = "Dependency: order_cache_tier_1 -> Order Service -> Lead Backend Engineer"

            # Construct Prompt
            prompt_template = PromptTemplate.from_template(
                """You are a site reliability engineering AI. 
                Use the following Vector Context and Graph Context to answer the user's question accurately.
                Keep the answer concise and direct.
                
                Vector Context: {vector_context}
                Graph Context: {graph_context}
                
                Question: {question}
                
                Answer:"""
            )
            formatted_prompt = prompt_template.format(
                vector_context=vector_context,
                graph_context=graph_context,
                question=user_question
            )

            # Invoke LLM
            final_answer = LLM.invoke(formatted_prompt)
            
            # Display Text Output
            st.markdown(final_answer.content)
            
            # Generate and Play Voice Output
            audio_file_path = generate_audio_file(final_answer.content)
            st.audio(audio_file_path, format="audio/wav")
            
            # Add to history
            st.session_state.chat_history.append({"role": "assistant", "content": final_answer.content})