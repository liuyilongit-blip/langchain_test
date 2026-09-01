import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()
# 初始化嵌入模型
embeddings = OpenAIEmbeddings(model="BAAI/bge-m3")
# 初始化向量存储
vectorstore = PineconeVectorStore(index_name="langchain-doc-index",embedding=embeddings)
# 初始化聊天模型
model = init_chat_model("deepseek-ai/DeepSeek-V4-Flash",model_provider="openai")

@tool(response_format="content_and_artifact")
def retrieve_context(query:str):
    """检索相关文档来帮助回答用户关于 LangChain 的问题"""
    #检索相似度最高的 4 个文档
    retrieved_docs = vectorstore.as_retriever().invoke(query,k=4)
    
    # 序列化文档
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    # 返回序列化的内容和原始文档
    return serialized, retrieved_docs

def run_llm(query: str) -> Dict[str, Any]:
    """
    运行 RAG 流程，利用检索到的文档回答查询。
    
    Args:
        query: 用户的提问
        
    Returns:
        包含以下内容的字典:
            - answer: 生成的回答
            - context: 检索到的文档列表
    """
    # 创建带有检索工具的 Agent
    system_prompt = (
        "你是一个乐于助人的 AI 助手，专门回答关于 LangChain 文档的问题。"
        "你可以使用一个能够检索相关文档的工具。"
        "在回答问题之前，请务必先使用该工具查找相关信息。"
        "回答时请始终注明并引用你所参考的文档来源。"
        "如果在检索到的文档中找不到答案，请如实告知用户。"
    )
    
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
    
    # 构建消息列表
    messages = [{"role": "user", "content": query}]
    
    # 调用 Agent
    response = agent.invoke({"messages": messages})
    
    # 从最后一条 AI 消息中提取回答
    answer = response["messages"][-1].content
    
    # 从 ToolMessage 的 artifact 中提取上下文文档
    context_docs = []
    for message in response["messages"]:
        # 检查是否为带有 artifact 的 ToolMessage
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # artifact 应包含 Document 对象列表
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="什么是深度 langchain？")
    print(result)
    