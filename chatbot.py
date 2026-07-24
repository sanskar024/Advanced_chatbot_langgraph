import json
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START,END
from typing import TypedDict
from langgraph.graph.message import BaseMessage, add_messages
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.prebuilt import tool_condition,ToolNode

import sqlite3
load_dotenv()
API_KEY = os.getenv("API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.2,
)

conn=sqlite3.connect("chat.db",check_same_thread=False)



class BotState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

graph=StateGraph(BotState)

@tool
def calculator(  a:float,b:float,operator:  str ):
    """this tool takes 2 no and perfrom calculation based on given operator"""
    if(operator=="+"):
        return a+b
    elif(operator=="-"):
        return a-b
    elif(operator=="*"):
        return a*b
    elif(operator=="/" ):
        
 

def chat_node(State:BotState):
    messages=State["messages"]
    response=llm.invoke(messages)
    return {"messages": response}
   

graph.add_node("chat_node",chat_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edge("chat_node",tool_condition )

checkpointer=SqliteSaver(conn=conn)

workflow=graph.compile(checkpointer=checkpointer)

def fetch_threads():
    all_threads = set()

    for checkpoint in checkpointer.list(None):
        all_threads.add(
            checkpoint.config["configurable"]["thread_id"]
        )

    return list(all_threads)