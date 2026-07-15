import streamlit as st
from chatbot import workflow
from langchain_core.messages import HumanMessage, BaseMessage


if("messages" not in st.session_state):
    st.session_state["messages"]=[]

# for printing the messages in the chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def response_generator():
    for message_chunk, metadata in workflow.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": "t1"}},
        stream_mode="messages",
    ):
        if message_chunk.content:
            yield message_chunk.content

user_input = st.chat_input("Enter your query:")
if user_input:
    with st.chat_message("user"):
        st.text(user_input)
    st.session_state["messages"].append({"role":"user","content":user_input})

    with st.chat_message("assistant"):
        ai_message = st.write_stream(response_generator)
    st.session_state["messages"].append({"role":"assistant","content":ai_message})
