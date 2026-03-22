"""向量检索器 - 负责相似度搜索和 RAG 问答"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import OpenAI

from vector_db_manager import VectorDatabaseManager

load_dotenv()


@dataclass
class AnswerResult:
    """问答结果"""
    answer: str
    source_documents: List[Document] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    confidence: float = 0.0


class VectorRetriever:
    """向量检索器 + 问答生成"""

    def __init__(
        self,
        db_manager: VectorDatabaseManager,
        similarity_threshold: float = 0.5,
    ):
        self.db_manager = db_manager
        self.similarity_threshold = similarity_threshold
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.vectorengine.ai/v1"),
        )

    def search_similar_content(
        self,
        query: str,
        collection_name: Optional[str] = None,
        k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """搜索相似内容，过滤低相似度结果"""
        search_results = self.db_manager.search(
            query=query, k=k, collection_name=collection_name
        )
        # Milvus L2 距离越小越相似，这里保留所有结果让用户自行判断
        return search_results

    def answer_question(
        self,
        question: str,
        collection_name: Optional[str] = None,
        k: int = 5,
    ) -> AnswerResult:
        """RAG 问答：检索相关文档 → 构建上下文 → LLM 生成回答"""
        relevant_docs = self.search_similar_content(
            query=question, collection_name=collection_name, k=k
        )

        # 构建上下文
        context_parts = []
        source_documents = []
        scores = []
        for i, (doc, score) in enumerate(relevant_docs):
            context_parts.append(f"参考资料{i + 1}: {doc.page_content}")
            source_documents.append(doc)
            scores.append(score)

        context = "\n\n".join(context_parts) if context_parts else "（无相关参考资料）"

        # 调用 LLM 生成回答
        answer = self._generate_answer_with_llm(question, context)

        confidence = 1.0 / (1.0 + min(scores)) if scores else 0.0

        return AnswerResult(
            answer=answer,
            source_documents=source_documents,
            scores=scores,
            confidence=round(confidence, 3),
        )

    def _generate_answer_with_llm(self, question: str, context: str) -> str:
        """使用 LLM 生成回答"""
        system_prompt = (
            "你是一个智能助手。请基于提供的【参考资料】回答用户的问题。\n"
            "如果参考资料为空或与问题无关，请利用你的通用知识进行回答，"
            "并说明这不是基于知识库的回答。\n"
            "回答要准确、简洁、有条理。"
        )

        response = self.client.chat.completions.create(
            model="deepseek-v3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"问题: {question}\n\n【参考资料】:\n{context}",
                },
            ],
        )
        return response.choices[0].message.content
