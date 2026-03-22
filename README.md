# RAG 知识库助手

基于 FAISS + LangChain + OpenAI 兼容 API 的本地知识库问答系统，支持文档上传、向量检索、智能问答，附带 Web 前端界面。

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 向量数据库 | FAISS | 本地文件存储，零配置 |
| 嵌入模型 | text-embedding-ada-002 | 通过 OpenAI 兼容 API |
| 大语言模型 | deepseek-v3.2 | 通过 OpenAI 兼容 API |
| 应用框架 | LangChain | RAG 流水线 |
| Web 框架 | Flask | REST API + 前端 |
| 文档处理 | PyPDF / Docx2txt / Pandas | PDF、TXT、DOCX、CSV、Excel |

## 项目结构

```
rag-knowledge-assistant/
├── server.py               # Flask 服务入口（端口 5001）
├── index.html              # Web 前端界面
├── api_integration.py      # REST API 蓝图
├── vector_db_manager.py    # 核心：FAISS 向量数据库管理
├── vector_retriever.py     # 向量检索 + LLM 问答
├── document_loader.py      # 多格式文档加载器
├── query_system.py         # 命令行查询工具
├── upload_document.py      # 命令行上传工具
├── faiss_index/            # FAISS 索引持久化目录（自动生成）
├── requirements.txt
├── .env
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
cd rag-knowledge-assistant
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```
COLLECTION_NAME=agent_rag
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=https://api.vectorengine.ai/v1
```

### 3. 启动服务

```bash
python3 server.py
```

浏览器打开 http://localhost:5001 即可使用 Web 界面。

## 使用方式

### Web 前端（推荐）

访问 http://localhost:5001：
- 拖拽或点击上传文档（TXT / PDF / DOCX / CSV / XLSX）
- 输入问题回车提问
- 查看 AI 回答、置信度、参考来源

### API 调用

```bash
# 上传文档（文件路径）
curl -X POST http://localhost:5001/api/vector/upload_document \
  -H "Content-Type: application/json" \
  -d '{"file_path": "test_doc.txt"}'

# 上传文档（文件流）
curl -X POST http://localhost:5001/api/vector/upload_file \
  -F "file=@test_doc.txt"

# RAG 问答
curl -X POST http://localhost:5001/api/vector/query \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG？", "k": 3}'

# 纯相似度搜索
curl -X POST http://localhost:5001/api/vector/search \
  -H "Content-Type: application/json" \
  -d '{"query": "向量数据库", "k": 5}'

# 查看知识库信息
curl http://localhost:5001/api/vector/collection_info

# 清空知识库
curl -X POST http://localhost:5001/api/vector/clear_collection
```

### 命令行工具

```bash
# 上传文档（修改 upload_document.py 中的 FILE_PATH）
python3 upload_document.py

# 查询问答（修改 query_system.py 中的 QUESTION）
python3 query_system.py
```

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 前端界面 |
| `/api/vector/upload_document` | POST | 上传文档（路径方式） |
| `/api/vector/upload_file` | POST | 上传文档（文件流） |
| `/api/vector/query` | POST | RAG 问答 |
| `/api/vector/search` | POST | 相似度搜索 |
| `/api/vector/collection_info` | GET | 知识库状态 |
| `/api/vector/clear_collection` | POST | 清空知识库 |

## 工作原理

```
文档上传流程：文档 → 切片(500字/块) → Embedding → FAISS 索引存储
问答检索流程：问题 → Embedding → FAISS 相似搜索 → Top-K 结果 → LLM 生成回答
```

## 备注

- 端口使用 5001（macOS 5000 被 AirPlay Receiver 占用）
- FAISS 索引自动持久化到 `faiss_index/` 目录，重启不丢失
- 原文档设计使用 Milvus，因 macOS ARM gRPC 兼容问题改用 FAISS
