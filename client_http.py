# client_http.py


import json
import requests


# LangChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
load_dotenv()

# 初始化 ChatOpenAI 模型
model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL"),
)

server_url = "http://127.0.0.1:8000"

def fetch_tools():
    resp = requests.get(f"{server_url}/tools")
    all_items = resp.json()
    return all_items, all_items  # 资源与工具统一列表

def ask_llm_to_select_tool_and_resource(user_question, tools, resources):
    tools_desc = "\n".join([
        f"- {t['name']}: {t.get('desc', t.get('description', ''))} (method: {t.get('method', 'GET')}, url: {t.get('url', '')})"
        for t in tools
    ])

    resources_desc = "\n".join([
        f"- {r['name']}: {r.get('desc', r.get('description', ''))} (url: {r.get('url', '')})"
        for r in resources
    ])


    system_prompt = f'''
你是一个智能助手，用户提出问题后，你首先考虑要不要使用工具列表中和资源列表，选择合适的工具进行调用，
并选择是否需要配合资源。你的输出应为一个 JSON，对应字段如下：
tool: {{ name: 工具名, url: 工具路径, method: 请求方式, params: {{参数}} }},
resource: {{ name: 资源名, desc: 简要说明 }}。
请不要随意改变工具的url。
如果不需要资源，resource 设置为 null。
仅使用以下工具列表：
{tools_desc}

以下是可用资源：
{resources_desc}
'''


    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ]
    print("📤 提示词已发送到 LLM，正在生成回答...")
    response = model.invoke(messages)
    print("📥 LLM 返回内容：", response.content)

    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        print("❌ LLM 返回格式错误")
        raise

def call_tool(tool):
    url = f"{server_url}{tool.get('url')}"
    method = tool.get("method", "GET").upper()
    params = tool.get("params", {})

    print(f"👉 请求 URL: {url}")
    print(f"👉 请求方法: {method}")
    print(f"👉 请求参数: {params}")

    if method == "GET":
        response = requests.get(url, params=params)
    else:
        raise NotImplementedError(f"仅支持 GET 方法，当前为 {method}")

    try:
        return response.json()["result"]
    except Exception as e:
        print("❌ 工具调用异常，返回内容：", response.text)
        raise e

def summarize_answer(question, raw_result):
    messages = [
        SystemMessage(content="请根据调用返回的原始信息，总结出一个简明清晰的中文答案。"),
        HumanMessage(content=f"问题：{question}\n调用结果：{raw_result}")
    ]
    response = model.invoke(messages)
    return response.content

def main():
    user_question = input("请输入问题：")
    # user_question = "1+2=?"
    print("Step 1: 获取工具和资源...")
    tools, resources = fetch_tools()

    max_retries = 3
    for attempt in range(max_retries):
        print(f"\nStep 2: 使用 LLM 选择工具和资源...（第 {attempt + 1} 次尝试）")
        try:
            llm_selection_json = ask_llm_to_select_tool_and_resource(user_question, tools, resources)
            tool = llm_selection_json["tool"]
            resource = llm_selection_json.get("resource")

            print(f"Step 3: 使用工具 {tool['name']} 调用接口 {tool['url']} ...")
            result = call_tool(tool)

            print("Step 4: 使用 LLM 总结答案...")
            summary = summarize_answer(user_question, result)
            print("\n✅ 最终答案：", summary)
            break  # 成功则跳出循环

        except Exception as e:
            print(f"⚠️ 第 {attempt + 1} 次执行失败：{e}")
            if attempt < max_retries - 1:
                print("🧠 向 LLM 反馈错误，尝试生成新的调用方案...")
                messages = [
                    SystemMessage(content="你刚才输出的 JSON 无法成功调用接口，返回了错误信息。请基于以下错误信息，修复输出。"),
                    HumanMessage(content=f"用户问题：{user_question}\n错误信息：{str(e)}")
                ]
                response = model.invoke(messages)
                print("📥 LLM 修复输出内容：", response.content)
                try:
                    llm_selection_json = json.loads(response.content)
                except:
                    print("❌ 修复输出仍无法解析为 JSON，将重新尝试...")
            else:
                print("❌ 多次尝试均失败，程序退出。")

if __name__ == "__main__":
    main()
