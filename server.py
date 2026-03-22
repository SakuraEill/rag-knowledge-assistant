"""Flask 服务器入口"""

import os
import numpy as np
from flask import Flask, send_from_directory
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from dotenv import load_dotenv

from api_integration import vector_bp

load_dotenv()


class NumpyJSONProvider(DefaultJSONProvider):
    """支持 numpy 类型的 JSON 序列化"""
    @staticmethod
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return DefaultJSONProvider.default(o)


app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)
app.config["JSON_AS_ASCII"] = False  # 返回中文而非 Unicode 转义
CORS(app)

# 注册蓝图
app.register_blueprint(vector_bp)


@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__) or ".", "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
