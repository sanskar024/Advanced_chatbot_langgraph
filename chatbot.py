import json
from dotenv import load_dotenv
import requests
from langsmith import wrappers
from langchain_core.tools import tool
from langchain.agents import create_agent
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

thread='thread_1'

while True:
    user_input=input("Enter your query: ")
    if(user_input.lower().strip(" ")=="exit"):
        break
    history= workflow.invoke({"messages":[HumanMessage(content=user_input)]}, config={"configurable": {"thread_id": thread}})
    print("Bot:",history["messages"][-1].content)