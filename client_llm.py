# client_llm.py
'''
对于支持 function call 的模型（如 OpenAI、通义千问、llama）有效，
对于不支持 function call 的模型（如 deepseek）无效。
'''

import asyncio
import os
from dotenv import load_dotenv

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage

# 加载 .env 环境变量
load_dotenv()

# 初始化 ChatOpenAI 模型
model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),  # 如 "qwen-plus"
)


async def run_agent(user_question: str):
    async with streamablehttp_client("http://127.0.0.1:8000/mcp/") as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # 加载服务端定义的工具和资源
            tools = await load_mcp_tools(session)
            print("加载到的工具/资源：")
            for tool in tools:
                print(f"- {tool.name}: {tool.description}")

            # 创建并运行 ReAct Agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": user_question})

            # 提取最终 AI 回复的内容
            for message in reversed(agent_response["messages"]):
                if isinstance(message, AIMessage) and message.content:
                    return message.content
                if isinstance(message, ToolMessage) and message.content:
                    return message.content
            return "未获取到有效回答。"


if __name__ == "__main__":
    while True:
        user_question = input("请输入问题（输入 exit 退出）：")
        if user_question.strip().lower() == "exit":
            print("已退出。")
            break

        result = asyncio.run(run_agent(user_question))
        print(result)
