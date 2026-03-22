"""向量数据库管理器 - 核心模块，负责文档处理、向量存储和检索

使用 FAISS 作为本地向量数据库，无需 Docker 或外部服务。
也可切换为 Milvus（需要 Docker）。
"""

import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from document_loader import DocumentLoader

load_dotenv()

# FAISS 持久化目录
FAISS_INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")


class VectorDatabaseManager:
    """向量数据库管理器（FAISS 本地模式）"""

    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "agent_rag")
        self._init_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
        self.doc_loader = DocumentLoader()
        self.vectorstore = None
        self._load_existing_index()

    def _init_embeddings(self):
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.vectorengine.ai/v1"),
            model="text-embedding-ada-002",
        )

    def _index_path(self, collection_name: Optional[str] = None) -> str:
        name = collection_name or self.collection_name
        return os.path.join(FAISS_INDEX_DIR, name)

    def _load_existing_index(self):
        """尝试加载已有的 FAISS 索引"""
        path = self._index_path()
        if os.path.exists(path):
            try:
                self.vectorstore = FAISS.load_local(
                    path, self.embeddings, allow_dangerous_deserialization=True
                )
                print(f"✅ 已加载 FAISS 索引: {self.collection_name}")
            except Exception as e:
                print(f"⚠️ 加载索引失败: {e}")
                self.vectorstore = None
        else:
            print(f"📂 索引不存在，等待上传文档: {self.collection_name}")

    def load_document(self, file_path: str) -> List[Document]:
        return self.doc_loader.load(file_path)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)

    def add_documents_to_db(
        self, documents: List[Document], collection_name: Optional[str] = None
    ):
        """将文档添加到 FAISS 向量数据库"""
        target = collection_name or self.collection_name
        path = self._index_path(target)

        if os.path.exists(path):
            vs = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            vs.add_documents(documents)
            vs.save_local(path)
            print(f"📝 已追加 {len(documents)} 个文档块到 '{target}'")
        else:
            vs = FAISS.from_documents(documents=documents, embedding=self.embeddings)
            os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
            vs.save_local(path)
            print(f"🆕 已创建索引 '{target}' 并插入 {len(documents)} 个文档块")

        if target == self.collection_name:
            self.vectorstore = vs

    def process_file(self, file_path: str, collection_name: Optional[str] = None) -> bool:
        try:
            print(f"📄 正在处理文件: {file_path}")
            documents = self.load_document(file_path)
            print(f"  加载了 {len(documents)} 个文档")
            split_docs = self.split_documents(documents)
            print(f"  切分为 {len(split_docs)} 个文档块")
            self.add_documents_to_db(split_docs, collection_name)
            return True
        except Exception as e:
            print(f"❌ 处理文件失败: {e}")
            import traceback; traceback.print_exc()
            return False

    def search(
        self, query: str, k: int = 5, collection_name: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        target = collection_name or self.collection_name
        path = self._index_path(target)
        if not os.path.exists(path):
            return []
        vs = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        return vs.similarity_search_with_score(query, k=k)

    def get_database_info(self, collection_name: Optional[str] = None) -> dict:
        target = collection_name or self.collection_name
        path = self._index_path(target)
        if not os.path.exists(path):
            return {"exists": False, "collection_name": target}
        vs = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        return {
            "exists": True,
            "collection_name": target,
            "num_entities": vs.index.ntotal,
        }

    def clear_collection(self, collection_name: Optional[str] = None) -> bool:
        target = collection_name or self.collection_name
        path = self._index_path(target)
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
            self.vectorstore = None
            print(f"🗑️ 已删除索引: {target}")
            return True
        return False
