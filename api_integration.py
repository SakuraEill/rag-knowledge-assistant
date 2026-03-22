"""Flask API 蓝图 - 提供 REST API 端点"""

import os
import tempfile

from flask import Blueprint, request, jsonify

from vector_db_manager import VectorDatabaseManager
from vector_retriever import VectorRetriever

vector_bp = Blueprint("vector", __name__, url_prefix="/api/vector")

# 延迟初始化核心组件
vector_manager = None
vector_retriever = None


def _get_components():
    """延迟初始化，避免导入时就连接 Milvus"""
    global vector_manager, vector_retriever
    if vector_manager is None:
        vector_manager = VectorDatabaseManager()
        vector_retriever = VectorRetriever(db_manager=vector_manager)
    return vector_manager, vector_retriever


@vector_bp.route("/upload_document", methods=["POST"])
def upload_document():
    """上传并处理文档（通过文件路径）"""
    mgr, _ = _get_components()
    data = request.get_json()
    if not data or "file_path" not in data:
        return jsonify({"success": False, "error": "缺少 file_path 参数"}), 400

    file_path = data["file_path"]
    collection_name = data.get("collection_name")

    success = mgr.process_file(file_path, collection_name)
    if success:
        return jsonify({
            "success": True,
            "message": f"文档处理成功: {file_path}",
            "database_info": mgr.get_database_info(collection_name),
        })
    return jsonify({"success": False, "error": "文档处理失败"}), 500


@vector_bp.route("/upload_file", methods=["POST"])
def upload_file():
    """上传文件流"""
    mgr, _ = _get_components()
    if "file" not in request.files:
        return jsonify({"success": False, "error": "缺少文件"}), 400

    file = request.files["file"]
    collection_name = request.form.get("collection_name")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        success = mgr.process_file(tmp_path, collection_name)
        if success:
            return jsonify({
                "success": True,
                "message": f"文件处理成功: {file.filename}",
                "database_info": mgr.get_database_info(collection_name),
            })
        return jsonify({"success": False, "error": "文件处理失败"}), 500
    finally:
        os.unlink(tmp_path)


@vector_bp.route("/query", methods=["POST"])
def query_documents():
    """RAG 问答"""
    _, retriever = _get_components()
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"success": False, "error": "缺少 question 参数"}), 400

    question = data["question"]
    collection_name = data.get("collection_name")
    k = data.get("k", 5)

    result = retriever.answer_question(
        question=question, collection_name=collection_name, k=k
    )

    sources = []
    for doc, score in zip(result.source_documents, result.scores):
        sources.append({
            "content": doc.page_content[:200],
            "metadata": doc.metadata,
            "score": float(score),
        })

    return jsonify({
        "success": True,
        "question": question,
        "answer": result.answer,
        "confidence": float(result.confidence),
        "sources": sources,
    })


@vector_bp.route("/search", methods=["POST"])
def search_documents():
    """纯相似度搜索"""
    _, retriever = _get_components()
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"success": False, "error": "缺少 query 参数"}), 400

    query = data["query"]
    collection_name = data.get("collection_name")
    k = data.get("k", 5)

    results = retriever.search_similar_content(
        query=query, collection_name=collection_name, k=k
    )

    items = []
    for doc, score in results:
        items.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        })

    return jsonify({"success": True, "query": query, "results": items})


@vector_bp.route("/collection_info", methods=["GET"])
def collection_info():
    """获取集合信息"""
    mgr, _ = _get_components()
    collection_name = request.args.get("collection_name")
    info = mgr.get_database_info(collection_name)
    return jsonify({"success": True, "info": info})


@vector_bp.route("/clear_collection", methods=["POST"])
def clear_collection():
    """清空集合"""
    mgr, _ = _get_components()
    data = request.get_json()
    collection_name = data.get("collection_name") if data else None
    success = mgr.clear_collection(collection_name)
    if success:
        return jsonify({"success": True, "message": "集合已清空"})
    return jsonify({"success": False, "error": "集合不存在"}), 404
