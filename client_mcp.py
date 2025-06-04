from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import asyncio
from openai import AsyncOpenAI
import json
import re

# 配置通义大模型（兼容 OpenAI 接口）
OPENAI_API_KEY = "sk-7d48078fa897417c9cfa5cfa70d95f9a"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

llm = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

def extract_json_from_text(text: str):
    """
    从文本中尝试提取首个JSON对象。
    返回解析的dict或None。
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

async def main(user_question):
    async with streamablehttp_client("http://127.0.0.1:8000/mcp/") as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # 获取可用工具列表
            tools = await session.list_tools()
            tool_names = [name for name, _ in tools]

            # 构造工具描述提示
            tool_descs = "\n".join([f"{name}: {desc}" for name, desc in tools])
            print("可用工具列表：\n", tool_descs)

            # 构造提示词，告诉模型如何输出
            prompt = f"""根据以下工具列表回答问题。如果需要调用工具，请用JSON格式返回工具名称和参数，否则直接回答：

可用工具：
{tool_descs}

用户问题：{user_question}

响应示例：
调用工具: {{"tool": "工具名", "arguments": {{}}}}
直接回答: 你的回答内容
"""

            response = await llm.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )

            content = response.choices[0].message.content.strip()
            tool_call = extract_json_from_text(content)

            if tool_call is not None and "tool" in tool_call:
                print(f"模型推荐了工具调用：{tool_call}")

                # 调用工具
                tool_result = await session.call_tool(tool_call["tool"], tool_call.get("arguments", {}))
                tool_output = tool_result.content[0].text

                # 用工具结果再调用模型生成最终回答
                final_prompt = f"""用户问题：{user_question}
工具调用结果：{tool_output}
请根据以上信息给出完整自然语言回答。"""

                final_response = await llm.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": final_prompt}],
                )
                print("最终回答：", final_response.choices[0].message.content)
            else:
                # 模型直接回答，无需调用工具
                print("直接回答：", content)

if __name__ == "__main__":
    while True:
        user_question = input("请输入问题(输入 exit 或 quit 退出)：")
        if user_question in ("exit", "quit"):
            break
        asyncio.run(main(user_question))
