import asyncio
from typing import List
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool as LangchainTool


def wrap_tools_as_single_input(tools: List[LangchainTool]) -> List[LangchainTool]:
    """
    将 MCP 工具包装成只接受一个 input: str 参数的 Tool，以兼容 ZeroShotAgent。
    """
    wrapped_tools = []
    for tool in tools:
        wrapped_tool = LangchainTool(
            name=tool.name,
            description=tool.description,
            func=lambda x, f=tool.func: f(x),
            coroutine=(lambda x, f=tool.coroutine: f(x)) if tool.coroutine else None
        )
        wrapped_tools.append(wrapped_tool)
    return wrapped_tools


# 初始化模型（deepseek 不支持 function calling，所以我们用 zero-shot agent）
model = ChatOllama(
    base_url="http://192.168.66.26:11434",  # Ollama 本地服务地址
    model="deepseek-r1"
)

async def run_agent():
    async with streamablehttp_client("http://127.0.0.1:8000/mcp/") as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            # 初始化
            await session.initialize()

            # 加载 MCP 工具
            tools = await load_mcp_tools(session)

            # 修复工具结构：包装成单输入参数
            tools = wrap_tools_as_single_input(tools)

            # 创建自定义 LangChain Agent
            agent = initialize_agent(
                tools=tools,
                llm=model,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True  # 输出推理过程
            )

            # 执行推理
            question = "现在几点了"
            result = agent.invoke(f"请用中文回答：{question}")
            return result

if __name__ == "__main__":
    result = asyncio.run(run_agent())
    print("\nAgent 回复：", result)
