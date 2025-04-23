import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Custom CSS styling for dark mode + soft UI
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Title & intro
st.title("🧘 MindCare Chat")
st.caption("💬 A safe, private space to express yourself and feel heard.")

# Sidebar for model configuration
with st.sidebar:
    st.header("🌿 Mental Health Companion")
    selected_model = st.selectbox(
        "Select an AI Listener",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
    st.divider()
    st.markdown("### How I Can Support You")
    st.markdown("""
    - 😌 Calm your thoughts  
    - 😔 Talk through sadness  
    - 🌟 Find motivation again  
    - 💭 Simply share what's on your mind  
    """)
    st.divider()
    st.markdown("🧠 Note: This is not a substitute for professional therapy or crisis help.")

# Connect to Ollama LLM
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.5
)

# System prompt with supportive, empathetic tone
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a supportive and empathetic mental health assistant. "
    "Listen carefully, respond kindly, and encourage self-reflection. "
    "You are not a doctor or therapist. Avoid giving medical advice. Always reply in a caring tone and in English."
)

# Initialize chat history
if "message_log" not in st.session_state:
    st.session_state.message_log = [{
        "role": "ai", 
        "content": "Hello! I'm here for you. How are you feeling today? 💛"
    }]

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Type here to share your thoughts...")

# Helper functions
def generate_ai_response(prompt_chain):
    pipeline = prompt_chain | llm_engine | StrOutputParser()
    return pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# On user message
if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})

    with st.spinner("🧠 Thinking..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)

    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.rerun()
