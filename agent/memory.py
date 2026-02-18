"""
长期记忆模块
- FAISS 向量库存储事实
- 从初始记忆文件加载
- 支持检索和新增记忆
"""
import json
import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from agent.config import (
    FAISS_INDEX_PATH,
    MEMORIES_FILE,
    LONG_TERM_TOP_K,
    EMBED_PROVIDER,
    EMBED_MODEL,
)


def _get_embedding_function():
    """根据配置返回 embedding 函数"""
    if EMBED_PROVIDER == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    elif EMBED_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=EMBED_MODEL)
    elif EMBED_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBED_MODEL)
    else:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


class LongTermMemory:
    def __init__(self, index_path: str = FAISS_INDEX_PATH):
        self.index_path = index_path
        self.embeddings = _get_embedding_function()
        self.vectorstore: FAISS | None = None
        self._load_or_create()

    def _load_or_create(self):
        """加载现有索引或从初始记忆文件创建"""
        if os.path.exists(self.index_path):
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            docs = self._load_initial_memories()
            if docs:
                self.vectorstore = FAISS.from_documents(docs, self.embeddings)
                self.vectorstore.save_local(self.index_path)

    def _load_initial_memories(self) -> list[Document]:
        """从 raymond_memories.json 加载初始记忆"""
        if not os.path.exists(MEMORIES_FILE):
            return []

        with open(MEMORIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        for mem in data.get("memories", []):
            doc = Document(
                page_content=mem["fact"],
                metadata={
                    "topic": mem.get("topic", ""),
                    "confidence": mem.get("confidence", "medium"),
                    "source": "initial",
                },
            )
            docs.append(doc)
        return docs

    def search(self, query: str, top_k: int = LONG_TERM_TOP_K) -> list[str]:
        """根据查询检索相关记忆"""
        if not self.vectorstore:
            return []
        results = self.vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]

    def add_memory(self, fact: str, topic: str = ""):
        """添加一条新记忆"""
        doc = Document(
            page_content=fact,
            metadata={"topic": topic, "source": "conversation"},
        )
        if self.vectorstore:
            self.vectorstore.add_documents([doc])
        else:
            self.vectorstore = FAISS.from_documents([doc], self.embeddings)
        self.vectorstore.save_local(self.index_path)

    def get_all_count(self) -> int:
        """返回当前记忆总数"""
        if not self.vectorstore:
            return 0
        return len(self.vectorstore.docstore._dict)
