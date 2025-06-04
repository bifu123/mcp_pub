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




# 如果我想实现 服务端根据客户端传入的参数user来提取数据库中的该用户的工具列表，作为服务端动态构建工具，数据库用什么好，怎样实现？