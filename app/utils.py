"""Small helpers shared by the graph, tools and the Streamlit frontend."""
import ast
import json
from typing import Any, Optional


def message_text(content: Any) -> str:
    """Flatten message content to plain text. Gemini (and MCP tools) can
    return a list of content blocks instead of a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return "" if content is None else str(content)


def parse_tool_payload(content: Any) -> Optional[dict]:
    """Parse a ToolMessage body back into a dict (JSON first, then Python
    literal). Returns None if it isn't a dict."""
    text = message_text(content)
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(text)
        except (ValueError, SyntaxError):
            continue
        return value if isinstance(value, dict) else None
    return None
