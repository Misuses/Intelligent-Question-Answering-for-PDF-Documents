import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage


# 上传PDF文档并生成向量数据库
def upload_and_process_pdf(pdf_path):
    # 加载PDF文档
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 将文档拆分为小块
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(docs)

    # 将文档存储到向量数据库
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embeddings,
        persist_directory="test_db",  # 本地保存路径
    )
    return vectorstore


# 用户提问并检索或生成答案
def rag_answer(question, vectorstore):
    # 从向量数据库中检索相关文档
    retriever = vectorstore.as_retriever(k=3)
    documents = retriever.invoke(question)

    # 如果找到相关文档，生成答案
    if documents:
        docs_txt = "\n\n".join(doc.page_content for doc in documents)
        rag_prompt = f"""您是一个用于问答任务的助手。以下是回答问题时要使用的上下文：

        {docs_txt}

        现在，审阅用户问题：

        {question}

        请提供该问题的答案。最多使用四句话，保持答案简洁。"""

        # 使用模型生成答案
        model = ChatOllama(model="llama3.2:3b")
        response = model.invoke([HumanMessage(content=rag_prompt)])
        return response.content
    else:
        # 如果没有找到相关文档，使用模型进行推理回复

        model = ChatOllama(model="llama3.2:3b")
        # 直接使用用户的问题生成答案
        response = model.invoke([HumanMessage(content=question)])
        return response.content