from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

# Load API key (only locally)
load_dotenv()  
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not set. Add it to your deployment secrets!")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0.7,
    google_api_key=api_key,
    request_timeout=60  # Avoid timeouts
)

# Define the LangGraph flow
class ChatState(BaseMessage, dict): pass  # simplified placeholder

def chat_node(state):
    messages = state.get("messages", [])
    response = llm.invoke({"messages": messages})
    return {"messages": [response]}

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
compiled_graph = graph.compile(checkpointer=InMemorySaver())
