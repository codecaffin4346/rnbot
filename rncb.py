import streamlit as st
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Load FAQ data
with open("reliefnet_faq_100.json", "r") as f:
    faq_data = json.load(f)

# Create a dictionary for fast lookup
faq_dict = {item["question"]: item for item in faq_data}
faq_questions = list(faq_dict.keys())

# Streamlit UI
st.set_page_config(page_title="ReliefNet Bot", layout="centered")
st.title("🤖 ReliefNet_bot")
st.subheader("Ask your question.")

# Initialize memory and Gemini model
if "conversation" not in st.session_state:
    memory = ConversationBufferMemory(return_messages=True)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    st.session_state.conversation = {"llm": llm, "memory": memory}

# Input from user
user_input = st.chat_input("Ask a question...")

# Display chat history
if st.session_state.conversation["memory"].chat_memory.messages:
    for msg in st.session_state.conversation["memory"].chat_memory.messages:
        role = "🤖" if msg.type == "ai" else "🧑"
        with st.chat_message(role):
            st.markdown(msg.content)

# Function to find closest FAQ match (naive string match or add semantic later)
def find_best_match(user_question):
    for question in faq_questions:
        if user_question.lower() in question.lower() or question.lower() in user_question.lower():
            return faq_dict[question]
    return None

# Generate reply from Gemini with system prompt
def generate_reply(user_question):
    matched_faq = find_best_match(user_question)
    if matched_faq:
        system_prompt = f"""
You are a professional, compassionate assistant working for ReliefNet — a care-focused platform offering mental health support, home nursing, postpartum services, and NGO volunteering.

You are grounded in ReliefNet’s knowledge base of FAQs. Use the provided FAQ below as your main reference.

If the answer is not directly available in the FAQ, use your best judgment to generate a helpful, accurate response — staying within the domain and tone of ReliefNet. You may creatively extend the idea, but do not mention that you're guessing or that the info isn't available.

Your tone should be:
- Warm, professional, and supportive
- Trustworthy, like a trained ReliefNet staff member
- Clear and simple, as if speaking to someone new to the platform

DO NOT:
- Suggest searching online
- Mention that you're an AI or language model
- Say "based on the provided text"
- Say "I'm not sure"

FAQ Category: {matched_faq['category']}
Q: {matched_faq['question']}
A: {matched_faq['answer']}

Now, answer the user's question in a way that is helpful and friendly. Stay aligned with ReliefNet's offerings, even if the answer isn't directly in the FAQ.
"""
    else:
        system_prompt = """
You are a compassionate and knowledgeable ReliefNet assistant. Answer the user's question in a warm, professional tone that aligns with ReliefNet's offerings (mental health support, home nursing, postpartum care, NGO volunteering). Be clear, kind, and helpful.
"""

    llm = st.session_state.conversation["llm"]
    memory = st.session_state.conversation["memory"]

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question),
    ]
    response = llm.invoke(messages)
    memory.chat_memory.add_user_message(user_question)
    memory.chat_memory.add_ai_message(response.content)
    return response.content

# Handle user input
if user_input:
    with st.chat_message("🧑"):
        st.markdown(user_input)

    reply = generate_reply(user_input)

    with st.chat_message("🤖"):
        st.markdown(reply)