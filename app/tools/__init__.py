from .calculator import calculator
from .mcp import load_mcp_tools
from .rag import rag_tool
from .stock import get_stock_price
from .web_search import web_search

mcp_tools = load_mcp_tools()

all_tools = [web_search, get_stock_price, calculator, rag_tool, *mcp_tools]
