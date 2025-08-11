from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st
from typing import TypedDict, Annotated

# Try importing st only if running in Streamlit
try:
    import streamlit as st
    STREAMLIT = True
except ImportError:
    STREAMLIT = False

load_dotenv()

def get_api_key():
    if STREAMLIT and "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return os.getenv("GOOGLE_API_KEY")

GOOGLE_API_KEY = get_api_key()

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY,
    request_timeout=60
)



class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state.get("messages", [])
    response = model.invoke(messages)  # pass directly
    return {"messages": [response]}


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=InMemorySaver())








