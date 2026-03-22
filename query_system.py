"""独立查询系统 - 命令行 RAG 问答"""

from dotenv import load_dotenv
from vector_db_manager import VectorDatabaseManager
from vector_retriever import VectorRetriever

load_dotenv()

# ========== 配置区域 ==========
QUESTION = "什么是RAG？"       # 修改为你的问题
COLLECTION_NAME = "agent_rag"  # 集合名称
TOP_K = 5                      # 返回结果数量
# ==============================


def main():
    print("=" * 50)
    print("🔍 RAG 知识库 - 查询系统")
    print("=" * 50)

    manager = VectorDatabaseManager()
    retriever = VectorRetriever(db_manager=manager)

    print(f"\n❓ 问题: {QUESTION}")
    print("-" * 50)

    result = retriever.answer_question(
        question=QUESTION,
        collection_name=COLLECTION_NAME,
        k=TOP_K,
    )

    print(f"\n💡 回答:\n{result.answer}")
    print(f"\n📊 置信度: {result.confidence}")

    if result.source_documents:
        print(f"\n📚 参考来源 ({len(result.source_documents)} 条):")
        for i, (doc, score) in enumerate(
            zip(result.source_documents, result.scores)
        ):
            print(f"  [{i + 1}] 相似度分数: {score:.4f}")
            print(f"      内容: {doc.page_content[:100]}...")


if __name__ == "__main__":
    main()
