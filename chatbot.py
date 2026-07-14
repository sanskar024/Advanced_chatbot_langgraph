import json
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START,END
from typing import TypedDict
from langgraph.graph.message import BaseMessage, add_messages
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, BaseMessage
load_dotenv()
API_KEY = os.getenv("API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.2,
)


class BotState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

graph=StateGraph(BotState)

def chat_node(State:BotState):
    messages=State["messages"]
    response=llm.invoke(messages)
    return {"messages": response}
   

graph.add_node("chat_node",chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
checkpoint=MemorySaver()
workflow=graph.compile(checkpointer=checkpoint)

