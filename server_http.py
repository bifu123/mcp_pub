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


