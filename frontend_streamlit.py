import uuid

import streamlit as st
from chatbot import workflow
from langchain_core.messages import HumanMessage, BaseMessage


# Session state

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "thread_ids" not in st.session_state:
    st.session_state["thread_ids"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())
    st.session_state["thread_ids"].append(st.session_state["thread_id"])


# Helpers

def thread_id():
    return str(uuid.uuid4())


def new_chat():
    new_thread_id = thread_id()

    st.session_state["thread_ids"].append(new_thread_id)
    st.session_state["thread_id"] = new_thread_id
    st.session_state["messages"] = []

    st.rerun()


def response_generator():
    for message_chunk, metadata in workflow.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={
            "configurable": {
                "thread_id": st.session_state["thread_id"]
            }
        },
        stream_mode="messages",
    ):
        if message_chunk.content:
            yield message_chunk.content


def load_messages(thread_id):
    return workflow.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    ).values.get("messages", [])


# Chat history

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# User input

user_input = st.chat_input("Enter your query:")

if user_input:
    with st.chat_message("user"):
        st.text(user_input)

    st.session_state["messages"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("assistant"):
        ai_message = st.write_stream(response_generator)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )


# Sidebar

st.sidebar.title("Chatbot")

if st.sidebar.button("New Chat"):
    new_chat()

for thread_id in st.session_state.get("thread_ids", [])[::-1]:

    if st.sidebar.button(f"Thread ID: {thread_id}"):

        st.session_state["thread_id"] = thread_id

        messages = load_messages(thread_id)
        temp_msg = []

        for msg in messages:

            if isinstance(msg, HumanMessage):
                temp_msg.append(
                    {
                        "role": "user",
                        "content": msg.content
                    }
                )

            else:
                temp_msg.append(
                    {
                        "role": "assistant",
                        "content": msg.content
                    }
                )

        st.session_state["messages"] = temp_msg