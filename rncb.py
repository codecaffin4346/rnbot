import streamlit as st
import os
import json
from dotenv import load_dotenv

# swap in Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load environment variables (including GOOGLE_API_KEY)
load_dotenv()

# Load the FAQ knowledge base
with open("reliefnet_faq_100.json", "r") as f:
    faq_data = json.load(f)

faq_dict = {item["question"]: item["answer"] for item in faq_data}

st.set_page_config(page_title="Reliefnet_bot", layout="centered")
st.title("🤖 Reliefnet_bot")
st.subheader("Ask your question from the FAQ or type your own.")

if "conversation" not in st.session_state:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    memory = ConversationBufferMemory(return_messages=True)
    st.session_state.conversation = ConversationChain(llm=llm, memory=memory)

# render past chat
if st.session_state.conversation.memory.chat_memory.messages:
    for msg in st.session_state.conversation.memory.chat_memory.messages:
        role = "🤖" if msg.type == "ai" else "🧑"
        with st.chat_message(role):
            st.markdown(msg.content)

user_input = st.chat_input("Ask a question...")

if user_input:
    if user_input in faq_dict:
        answer = faq_dict[user_input]
    else:
        answer = st.session_state.conversation.predict(input=user_input)

    with st.chat_message("🧑"):
        st.markdown(user_input)
    with st.chat_message("🤖"):
        st.markdown(answer)
