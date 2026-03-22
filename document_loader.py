"""文档加载器 - 支持 PDF、TXT、DOCX、CSV、Excel 等多种格式"""

import os
from typing import List
from langchain_core.documents import Document


class DocumentLoader:
    """多格式文档加载器"""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv", ".xlsx", ".xls", ".md"}

    def load(self, file_path: str) -> List[Document]:
        """根据文件扩展名自动选择加载器"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext == ".txt" or ext == ".md":
            return self._load_text(file_path)
        elif ext == ".docx":
            return self._load_docx(file_path)
        elif ext == ".csv":
            return self._load_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            return self._load_excel(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}，支持: {self.SUPPORTED_EXTENSIONS}")

    def _load_pdf(self, file_path: str) -> List[Document]:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        return loader.load()

    def _load_text(self, file_path: str) -> List[Document]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return [Document(page_content=content, metadata={"source": file_path})]

    def _load_docx(self, file_path: str) -> List[Document]:
        import docx2txt
        content = docx2txt.process(file_path)
        return [Document(page_content=content, metadata={"source": file_path})]

    def _load_csv(self, file_path: str) -> List[Document]:
        import pandas as pd
        df = pd.read_csv(file_path)
        content = df.to_string(index=False)
        return [Document(page_content=content, metadata={"source": file_path})]

    def _load_excel(self, file_path: str) -> List[Document]:
        import pandas as pd
        df = pd.read_excel(file_path)
        content = df.to_string(index=False)
        return [Document(page_content=content, metadata={"source": file_path})]
