import asyncio
from mcp.server.fastmcp import FastMCP
from datetime import datetime
import os
import inspect
import requests  # 用于搜索工具

# 初始化 MCP 实例
mcp = FastMCP("ToolServer")

# 示例工具：获取当前时间
@mcp.tool(name="get_current_time", description="获取当前时间的 ISO 格式字符串")
def get_current_time() -> str:
    return datetime.now().isoformat()

# 示例工具：计算两个数字的和
@mcp.tool(name="add_numbers", description="计算两个数字的和")
def add_numbers(a: float, b: float) -> float:
    return a + b

# 示例资源：读取 yuanlong.txt 内容
@mcp.resource("data://yuanlong_story", description="读取包含元龙居士经历的 yuanlong.txt 文件内容")
def read_yuanlong_story() -> str:
    file_path = os.path.join(os.path.dirname(__file__), "yuanlong.txt")
    if not os.path.exists(file_path):
        return "yuanlong.txt 文件不存在。"
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# 将资源封装为工具
def wrap_resource_as_tool(resource_name: str, resource_func, description: str):
    if inspect.iscoroutinefunction(resource_func):
        async def async_tool_wrapper(*args, **kwargs):
            print(f"[调用异步工具] {resource_name}")
            return await resource_func(*args, **kwargs)
        async_tool_wrapper.__name__ = resource_func.__name__ + "_tool"
        return mcp.tool(name=resource_name, description=description)(async_tool_wrapper)
    else:
        def sync_tool_wrapper(*args, **kwargs):
            print(f"[调用同步工具] {resource_name}")
            return resource_func(*args, **kwargs)
        sync_tool_wrapper.__name__ = resource_func.__name__ + "_tool"
        return mcp.tool(name=resource_name, description=description)(sync_tool_wrapper)

# 异步注册所有资源为工具
async def register_all_resources_as_tools():
    resources = await mcp.list_resources()
    print(f"[资源数] 发现 {len(resources)} 个资源，尝试注册为工具")
    for res in resources:
        resource_name = getattr(res, "name", None)
        resource_func = getattr(res, "func", None)
        description = getattr(res, "description", "")
        if resource_name is None or resource_func is None:
            print(f"[跳过] 资源信息缺失: {res}")
            continue
        print(f"[注册] 资源 {resource_name} 封装成工具，描述：{description}")
        wrap_resource_as_tool(resource_name, resource_func, description)

# 新增：DuckDuckGo 搜索工具
@mcp.tool(name="ddg_search", description="通过 DuckDuckGo 搜索引擎查询信息，参数格式为 {'q': '搜索关键词'}")
def ddk_ddg_search(q: str) -> str:
    """
    简单的 DuckDuckGo 查询工具，用于返回网页摘要。
    """
    try:
        resp = requests.get("https://api.duckduckgo.com/", params={
            "q": q,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1
        }, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        # 提取摘要
        abstract = data.get("AbstractText") or data.get("Heading") or "无结果"
        return abstract.strip()
    except Exception as e:
        return f"搜索失败: {e}"

# 启动服务
if __name__ == "__main__":
    asyncio.run(register_all_resources_as_tools())
    mcp.run(transport="streamable-http")
