import streamlit as st
from chatbot import workflow
from langchain_core.messages import HumanMessage, BaseMessage


if("messages" not in st.session_state):
    st.session_state["messages"]=[]

# for printing the messages in the chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Enter your query:")
if user_input:
    st.session_state["messages"].append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.text(user_input)
    response= workflow.invoke({"messages":[HumanMessage(content=user_input)]}, config={"configurable": {"thread_id": "thread_1"}})
    ai_message=response["messages"][-1].content
    st.session_state["messages"].append({"role":"assistant","content":ai_message})
    with st.chat_message("assistant"):
        st.text(ai_message)