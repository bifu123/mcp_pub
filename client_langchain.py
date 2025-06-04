import requests
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType

import os
from dotenv import load_dotenv
load_dotenv()

# 配置你的工具服务地址
TOOLS_API_URL = "http://localhost:8000/tools"

# 调用工具时用 GET 请求，url 用工具定义里的 url 字段拼接
def call_tool_api(tool_def, params):
    base_url = "http://localhost:8000"
    url = base_url + tool_def["url"]
    try:
        resp = requests.get(url, params=params)  # GET请求，参数放query
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", "无返回结果")
    except Exception as e:
        return f"调用工具失败: {e}"

# 构建工具列表，func 接收字符串，尝试转json，失败则用输入字符串作为参数input
def build_tools_from_server():
    resp = requests.get(TOOLS_API_URL)
    resp.raise_for_status()
    tools_info = resp.json()

    tools = []
    for tool_def in tools_info:
        name = tool_def["name"]
        description = tool_def.get("description", "无描述")
        param_defs = tool_def.get("parameters", {})

        def make_func(tool_def_inner, param_defs_inner):
            def func(arg_str):
                import json
                try:
                    params = json.loads(arg_str)
                    if not isinstance(params, dict):
                        raise ValueError()
                except:
                    # 如果只需要一个参数，自动包装
                    if "input" in param_defs_inner:
                        params = {"input": arg_str}
                    elif "q" in param_defs_inner:
                        params = {"q": arg_str}
                    else:
                        return f"参数格式错误：{arg_str}"
                return call_tool_api(tool_def_inner, params)
            return func

        tools.append(
            Tool(
                name=name,
                func=make_func(tool_def, param_defs),
                description=description
            )
        )
    return tools


def main():
    tools = build_tools_from_server()
    llm = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model=os.getenv("MODEL"),
    )

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    print("🤖 智能工具代理启动，输入 exit 退出。")
    while True:
        user_input = input("你：")
        if user_input.strip().lower() in ("exit", "quit"):
            print("再见！")
            break
        try:
            result = agent.run(user_input)
            print("🤖 答案：", result)
        except Exception as e:
            print(f"❌ 出错了:", {e})

if __name__ == "__main__":
    main()
