import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load environment variables
load_dotenv()

# Load the FAQ knowledge base
with open("reliefnet_faq_100.json", "r") as f:
    faq_data = json.load(f)

# Prepare dictionary for FAQ lookup
faq_dict = {item["question"]: item["answer"] for item in faq_data}
faq_questions = list(faq_dict.keys())

# Page config
st.set_page_config(page_title="Reliefnet_bot", layout="centered")
st.title("🤖 Reliefnet_bot")
st.subheader("Ask your question from the FAQ or type your own.")

# Initialize memory and model
if "conversation" not in st.session_state:
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))
    memory = ConversationBufferMemory(return_messages=True)
    st.session_state.conversation = ConversationChain(llm=llm, memory=memory)

# Input from user
user_input = st.chat_input("Ask a question...")

# Chat message history
if st.session_state.conversation.memory.chat_memory.messages:
    for msg in st.session_state.conversation.memory.chat_memory.messages:
        role = "🤖" if msg.type == "ai" else "🧑"
        with st.chat_message(role):
            st.markdown(msg.content)

# Handle user query
if user_input:
    # If input is in FAQ, respond directly
    if user_input in faq_dict:
        answer = faq_dict[user_input]
    else:
        # Use GPT-4o if not directly matched
        answer = st.session_state.conversation.predict(input=user_input)

    with st.chat_message("🧑"):
        st.markdown(user_input)

    with st.chat_message("🤖"):
        st.markdown(answer)
