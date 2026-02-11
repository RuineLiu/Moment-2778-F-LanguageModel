import json
import os
import shutil  # 用于删除旧文件夹
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================= 配置区域 =================
DATA_FILE = "./dataset/raymond_finetune_data_clean.jsonl"
FAISS_PATH = "raymond_faiss_index"
LLM_MODEL = "qwen3:1.7b"
EMBED_MODEL = "qllama/bge-small-zh-v1.5"

CONTEXT_WINDOW_SIZE = 8192

def load_and_process_data(filepath):
    """
    读取并切分数据。
    """
    raw_documents = []
    print(f"正在加载数据: {filepath} ...")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                messages = data.get("messages", [])

                conversation_text = ""
                for msg in messages:
                    if msg['role'] == 'system':
                        continue

                    role_name = "User" if msg['role'] == 'user' else "Raymond"
                    conversation_text += f"{role_name}: {msg['content']}\n"

                if conversation_text:
                    doc = Document(
                        page_content=conversation_text,
                        metadata={"source": "chat_history"}
                    )
                    raw_documents.append(doc)
            except json.JSONDecodeError:
                continue

    print(f"原始对话片段加载完成: {len(raw_documents)} 条")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n", "。", "！", "？", " ", ""]
    )

    split_documents = text_splitter.split_documents(raw_documents)
    print(f"切分后文档数量: {len(split_documents)} 条")

    return split_documents


def get_vectorstore():
    """初始化或加载向量数据库"""
    embedding_function = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)

    if os.path.exists(FAISS_PATH):
        print("检测到现有向量索引，正在加载...")
        vectorstore = FAISS.load_local(
            FAISS_PATH,
            embedding_function,
            allow_dangerous_deserialization=True
        )
    else:
        print("正在创建新向量索引...")
        docs = load_and_process_data(DATA_FILE)
        vectorstore = FAISS.from_documents(docs, embedding_function)
        vectorstore.save_local(FAISS_PATH)
        print("向量索引创建完成！")

    return vectorstore


def chat_with_raymond():
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.7,
        num_ctx=CONTEXT_WINDOW_SIZE
    )

    template = """
    你现在是 Raymond。你是一个幽默、说话带有独特讽刺风格的人。
    请根据以下从历史聊天记录中检索到的参考风格（Context），回答用户的问题。

    如果参考内容与当前问题无关，请忽略参考内容，保持 Raymond 的人设自由发挥。

    [参考历史对话]
    {context}

    [当前对话]
    User: {question}
    Raymond:
    """

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n---\n".join([d.page_content for d in docs])

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print("\n" + "=" * 30)
    print(f"Raymond ({LLM_MODEL}) 已上线。输入 'exit' 退出。")
    print("=" * 30 + "\n")

    while True:
        user_input = input("你: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        print("Raymond: ", end="", flush=True)
        try:
            for chunk in rag_chain.stream(user_input):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n[Error] 生成回复时出错: {e}")
            print("建议：尝试降低 vectorstore 的 k 值，或检查显存是否不足。")


if __name__ == "__main__":

    chat_with_raymond()