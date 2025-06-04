对于更多人的来说，快速跑通一个最简单的DEMO是更为普遍的需求， 而MCP的官方项目（https://github.com/modelcontextprotocol/python-sdk）说明绕了半天，调试并不顺利，并不利于快速体验上手。基于本人需求，选取了mcp中服务端和客户端使用http传输的方式，纯python代码，其它的不做探究。以下是整理后的操练教程：

## 1. 安装 mcp
```bash
pip install mcp
```

## 2. 编写 mcp server
```python
# server.py

from mcp.server.fastmcp import FastMCP
from datetime import datetime

# 有状态服务端（维护会话状态）
mcp = FastMCP("StatefulServer")

# 定义一个工具，返回当前日期时间格式化字符串
@mcp.tool()
def get_current_time() -> str:
    return datetime.now().isoformat()

# 以流式方式启动服务
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

## 3. 运行服务端
python server.py # 直接运行
```
输出：
(mcp) E:\Project\mcp>python server.py
[32mINFO[0m:     Started server process [[36m17748[0m]
[32mINFO[0m:     Waiting for application startup.
[05/23/25 12:11:07] INFO     StreamableHTTP session manager started                      streamable_http_manager.py:109
[32mINFO[0m:     Application startup complete.
[32mINFO[0m:     Uvicorn running on [1mhttp://127.0.0.1:8000[0m (Press CTRL+C to quit)


mcp dev server.py  # 开发模式
输出：
(mcp) E:\Project\mcp>mcp dev server.py --with-editable .
Starting MCP inspector...
⚙️ Proxy server listening on port 6277
🔍 MCP Inspector is up and running at http://127.0.0.1:6274 🚀
打开 Inspector http://127.0.0.1:6274，将看到WEB调试页面



mcp dev server.py --with pandas --with numpy # 加载科学计算
mcp dev server.py --with-editable . # 挂载本地编辑器
```

### 3.1 不同状态的服务端
```python
from mcp.server.fastmcp import FastMCP

# 有状态对话式服务、保留会话记忆
mcp = FastMCP("Stateful")
mcp.run(transport="streamable-http")

# 无状态 HTTP 工具服务（返回 SSE 流）
mcp = FastMCP("Stateless", stateless_http=True)
mcp.run(transport="streamable-http")

# 无状态 HTTP 工具服务（返回标准 JSON）
mcp = FastMCP("Stateless", stateless_http=True, json_response=True)
mcp.run(transport="streamable-http")

# transport="streamable-http" 使用流式输出
```

### 3.2 同时运行多个服务器
```python
# echo.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="EchoServer", stateless_http=True)


@mcp.tool(description="A simple echo tool")
def echo(message: str) -> str:
    return f"Echo: {message}"
```

```python
# math1.py 如果像官方教程文件名为math.py将会与python内置函数同名出错
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="MathServer", stateless_http=True)


@mcp.tool(description="A simple add tool")
def add_two(n: int) -> int:
    return n + 2
```

```python
# main.py # 对官方代码作了更正
import contextlib
from fastapi import FastAPI
import echo
import math1


# Create a combined lifespan to manage both session managers
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(echo.mcp.session_manager.run())
        await stack.enter_async_context(math1.mcp.session_manager.run())
        yield


app = FastAPI(lifespan=lifespan)
app.mount("/echo", echo.mcp.streamable_http_app())
app.mount("/math", math1.mcp.streamable_http_app())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

执行：
python main.py


## 4. 客户端
```python
# client.py

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import asyncio

async def main():
    # 连接到 MCP 服务端
    async with streamablehttp_client("http://127.0.0.1:8000/mcp") as (
        read_stream,
        write_stream,
        _,
    ):
        # 创建客户端会话
        async with ClientSession(read_stream, write_stream) as session:
            # 初始化连接（必须）
            await session.initialize()

            # 调用服务端的 get_current_time 工具（无参数）
            tool_result = await session.call_tool("get_current_time", {})

            # print("当前时间：", tool_result)
            print("当前时间：", tool_result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
```

运行客户端
```bash
python client.py
```

输出：
(mcp) E:\Project\mcp>python client.py
当前时间： 2025-05-23T12:36:24.869402

## 5. 增加服务端工具
```python
# server.py
from mcp.server.fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("ToolServer")

@mcp.tool(name="get_current_time", description="获取当前时间的 ISO 格式字符串")
def get_current_time() -> str:
    return datetime.now().isoformat()

@mcp.tool(name="add_numbers", description="计算两个数字的和")
def add_numbers(a: float, b: float) -> float:
    return a + b


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

```python
# client.py
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import asyncio

async def main():
    async with streamablehttp_client("http://127.0.0.1:8000/mcp/") as (
        read_stream,
        write_stream,
        _,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("可用工具:")
            for name, description in tools:
                print(f"- {name}: {description}")

            resources = await session.list_resources()
            print("\n可用资源:")
            for name, description in resources:
                print(f"- {name}: {description}")



            # # 调用服务端的 get_current_time 工具（无参数）
            # tool_result = await session.call_tool("get_current_time", {})

            # # print("当前时间：", tool_result)
            # print("当前时间：", tool_result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())

```
(mcp) E:\Project\mcp>python client.py

可用工具:
- meta: None
- nextCursor: None
- tools: [Tool(name='get_current_time', description='获取当前时间的 ISO 格式字符串', inputSchema={'properties': {}, 'title': 'get_current_timeArguments', 'type': 'object'}, annotations=None), Tool(name='add_numbers', description='计算两个数字的和', inputSchema={'properties': {'a': {'title': 'A', 'type': 'number'}, 'b': {'title': 'B', 'type': 'number'}}, 'required': ['a', 'b'], 'title': 'add_numbersArguments', 'type': 'object'}, annotations=None)]

可用资源:
- meta: None
- nextCursor: None
- resources: []


## 6. 关于服务端资源
MCP的资源是不成熟的，所有定义的资源必须封装成工具让客户端去发现调用。可能它只是一种思想，即需要运算后得到结果的内容叫工具，而不需要或很少需要运算逻辑而得到的静态内容叫资源。但目前它支持非常不够好，我们还是略过吧。


## 7. 让大模型根据用户问题自主调用工具
### 7.1 支持 function call 的模型
```python
# clinet_llm.py
'''
对于支持function call的模型有用，比如openai、通义千问、llama对于不支持function call的deepseek是没有用的
'''

import asyncio
import os
from dotenv import load_dotenv

from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

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
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": user_question})

            # 提取最终 AI 回复的内容
            for message in reversed(agent_response["messages"]):
                if message.type == "ai" and message.content:
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

```

### 7.2 不支持 function call 的模型
```python
'''
如果不想动服务端，则可以修改客户端以使用langchain的agent tool：
'''
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
            result = agent.run(question)
            return result

if __name__ == "__main__":
    result = asyncio.run(run_agent())
    print("\nAgent 回复：", result)

```


## 8. 使用 http 来实现
```python
# server_http.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # 支持跨域

# ✅ 工具列表
@app.route("/tools", methods=["GET"])
def list_tools():
    return jsonify([
        {
            "name": "get_current_time",
            "description": "获取当前时间",
            "method": "GET",
            "url": "/tools/get_current_time"
        },
        {
            "name": "add_numbers",
            "description": "计算两个数的和，参数：a, b",
            "method": "GET",
            "url": "/tools/add"
        },
        {
            "name": "yuanlong_story",
            "description": "对元龙驿的介绍",
            "method": "GET",
            "url": "/tools/yuanlong_story"
        },
        {
            "name": "read_file",
            "description": "读取服务器本地文件内容（yuanlong.txt）",
            "method": "GET",
            "url": "/tools/read_file"
        }
    ])

# ✅ 实际工具接口
@app.route("/tools/get_current_time", methods=["GET"])
def get_current_time():
    return jsonify({"result": datetime.now().isoformat()})

@app.route("/tools/add", methods=["GET"])
def add_numbers():
    try:
        a = float(request.args.get("a"))
        b = float(request.args.get("b"))
        return jsonify({"result": a + b})
    except (TypeError, ValueError):
        return jsonify({"error": "参数错误，需要提供浮点数 a 和 b"}), 400

@app.route("/tools/yuanlong_story", methods=["GET"])
def yuanlong_story():
    return jsonify({
        "result": "元龙驿是贵州省贵定县元龙山的一个古代驿站。"
    })

@app.route("/tools/read_file", methods=["GET"])
def read_file():
    file_path = "yuanlong.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return jsonify({"result": f.read()})
    return jsonify({"result": "文件不存在"})

# ✅ 启动服务
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
```

```python
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
```

## 9. 使用 langchain 实现
```python
# server_langchain.py

from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# 网上搜索
@app.route("/tools/ddg_search", methods=["GET"])
def ddg_search():
    query = request.args.get("q") or request.args.get("input") or ""
    if not query:
        return jsonify({"error": "缺少查询参数 q 或 input"}), 400
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
            "skip_disambig": 1
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        abstract = data.get("AbstractText", "无摘要")
        return jsonify({"result": abstract})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 工具函数：获取当前时间
@app.route("/tools/get_current_time", methods=["GET"])
def get_current_time():
    from datetime import datetime
    return jsonify({"result": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

# 工具函数：两个数字相加
@app.route("/tools/add", methods=["GET"])
def add():
    try:
        a = float(request.args.get("a"))
        b = float(request.args.get("b"))
        return jsonify({"result": a + b})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 工具函数：返回元龙故事
@app.route("/tools/yuanlong_story", methods=["GET"])
def yuanlong_story():
    return jsonify({"result": "元龙居士名陈碧甫，清代著名学者..."})

# 工具函数：读取文件内容
@app.route("/tools/read_file", methods=["GET"])
def read_file():
    try:
        with open("yuanlong.txt", "r", encoding="utf-8") as f:
            return jsonify({"result": f.read()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 元数据接口：返回工具列表（结构化）
@app.route("/tools", methods=["GET"])
def get_tools():
    tool_list = [
        {
            "name": "get_current_time",
            "description": "获取当前时间，无需参数",
            "url": "/tools/get_current_time",
            "parameters": {}
        },
        {
            "name": "add_numbers",
            "description": "计算两个数的和，参数 a 和 b 是数字",
            "url": "/tools/add",
            "parameters": {
                "a": {"type": "number", "description": "第一个数字"},
                "b": {"type": "number", "description": "第二个数字"}
            }
        },
        {
            "name": "yuanlong_story",
            "description": "提供元龙居士的介绍，无需参数",
            "url": "/tools/yuanlong_story",
            "parameters": {}
        },
        {
            "name": "read_file",
            "description": "读取 yuanlong.txt 文件内容，无需参数",
            "url": "/tools/read_file",
            "parameters": {}
        },
        {
            "name": "ddg_search",
            "description": "使用 DuckDuckGo 搜索，参数 q 是查询字符串",
            "url": "/tools/ddg_search",
            "parameters": {
                "q": {"type": "string", "description": "搜索查询内容"}
            }
        },
    ]
    return jsonify(tool_list)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
```

```python
# server_langchain.py

from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# 网上搜索
@app.route("/tools/ddg_search", methods=["GET"])
def ddg_search():
    query = request.args.get("q") or request.args.get("input") or ""
    if not query:
        return jsonify({"error": "缺少查询参数 q 或 input"}), 400
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
            "skip_disambig": 1
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        abstract = data.get("AbstractText", "无摘要")
        return jsonify({"result": abstract})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 工具函数：获取当前时间
@app.route("/tools/get_current_time", methods=["GET"])
def get_current_time():
    from datetime import datetime
    return jsonify({"result": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

# 工具函数：两个数字相加
@app.route("/tools/add", methods=["GET"])
def add():
    try:
        a = float(request.args.get("a"))
        b = float(request.args.get("b"))
        return jsonify({"result": a + b})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 工具函数：返回元龙故事
@app.route("/tools/yuanlong_story", methods=["GET"])
def yuanlong_story():
    return jsonify({"result": "元龙居士名陈碧甫，清代著名学者..."})

# 工具函数：读取文件内容
@app.route("/tools/read_file", methods=["GET"])
def read_file():
    try:
        with open("yuanlong.txt", "r", encoding="utf-8") as f:
            return jsonify({"result": f.read()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 元数据接口：返回工具列表（结构化）
@app.route("/tools", methods=["GET"])
def get_tools():
    tool_list = [
        {
            "name": "get_current_time",
            "description": "获取当前时间，无需参数",
            "url": "/tools/get_current_time",
            "parameters": {}
        },
        {
            "name": "add_numbers",
            "description": "计算两个数的和，参数 a 和 b 是数字",
            "url": "/tools/add",
            "parameters": {
                "a": {"type": "number", "description": "第一个数字"},
                "b": {"type": "number", "description": "第二个数字"}
            }
        },
        {
            "name": "yuanlong_story",
            "description": "提供元龙居士的介绍，无需参数",
            "url": "/tools/yuanlong_story",
            "parameters": {}
        },
        {
            "name": "read_file",
            "description": "读取 yuanlong.txt 文件内容，无需参数",
            "url": "/tools/read_file",
            "parameters": {}
        },
        {
            "name": "ddg_search",
            "description": "使用 DuckDuckGo 搜索，参数 q 是查询字符串",
            "url": "/tools/ddg_search",
            "parameters": {
                "q": {"type": "string", "description": "搜索查询内容"}
            }
        },
    ]
    return jsonify(tool_list)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

```

```python
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

```

## 10. 结论
MCP目前的工具调用模型和 LangChain 的区别
- MCP的设计是：
大语言模型基于工具列表做一次决策，选择调用哪个工具（一次调用一个工具）。
你调用该工具，拿到结果后，再将结果返回给模型，让模型生成回答。
这是一个单步工具调用的流程，模型不会自动“连环”调用多个工具。

- LangChain 的 Agent（尤其是 ReAct、Multi-Tool agent）：
支持模型根据当前上下文和已有工具调用结果连续拆解任务。
模型可以自主决定调用多个工具，交替调用和观察结果，直到得到最终答案。
这是一个多步、动态交互式工具调用流程。

- 纯手工http实现，优点是全程可控可调，但是对于编写提示词让大模型进行复杂任务时，难以达到langchain专业的程度。

langchain仍然是大模型应用的巅峰。MCP值得肯定的一点：它提出了将工具作为服务，提高了共享和利用率，并提出一个标准接口，但是它对于多步任务拆解等复杂任务力不从心，资源也仅是给出理念而己，本质还是工具。它的完善或许需要时日。


## 11. 交流
QQ群：222302526


