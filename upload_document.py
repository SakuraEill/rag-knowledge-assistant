"""独立上传脚本 - 命令行上传文档到知识库"""

from dotenv import load_dotenv
from vector_db_manager import VectorDatabaseManager

load_dotenv()

# ========== 配置区域 ==========
FILE_PATH = "your_document.pdf"  # 修改为你的文件路径
COLLECTION_NAME = "agent_rag"    # 集合名称
# ==============================


def main():
    print("=" * 50)
    print("📚 RAG 知识库 - 文档上传工具")
    print("=" * 50)

    manager = VectorDatabaseManager()
    success = manager.process_file(FILE_PATH, COLLECTION_NAME)

    if success:
        info = manager.get_database_info(COLLECTION_NAME)
        print(f"\n✅ 上传成功！")
        print(f"   集合: {info['collection_name']}")
        print(f"   文档数: {info['num_entities']}")
    else:
        print("\n❌ 上传失败，请检查文件路径和 Milvus 连接")


if __name__ == "__main__":
    main()
