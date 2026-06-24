from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage,ToolMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm = ChatGroq(model='openai/gpt-oss-120b')

# MCP client for local FastMCP server
client = MultiServerMCPClient(
    {
        "arith": {
    "transport": "stdio",
    "command": "python3",
    "args": ["/home/sushant/Desktop/Gen Ai/LangGraph/LangGraph_chatbot/mcp-client-langgraph/math_server.py"],
},
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def build_graph():
    tools = await client.get_tools()
    print(tools)
    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        messages = state["messages"]
    
    # Sanitize empty tool message content (Groq requires non-empty)
        sanitized = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and not msg.content:
                msg = ToolMessage(
                    content="Tool executed successfully.",
                    tool_call_id=msg.tool_call_id
            )
            sanitized.append(msg)
    
        system = SystemMessage(content=
        "You are a helpful assistant. Use the available tools to answer questions accurately."
    )
        response = await llm_with_tools.ainvoke([system] + sanitized)
        return {'messages': [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    return graph.compile()

async def main():
    chatbot = await build_graph()
    result = await chatbot.ainvoke({"messages": [HumanMessage(content="Give me all my expenses for the month of Nov from 1 Nov to 30 Nov")]})
    print(result['messages'][-1].content)

if __name__ == '__main__':
    asyncio.run(main())