from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # Counts self-correction passes made *within the current user turn*.
    # The frontend resets this to 0 whenever it sends a fresh human message
    # (see frontend_streamlit.py) so it doesn't accumulate across turns.
    correction_attempts: int
