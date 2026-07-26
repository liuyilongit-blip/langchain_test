import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings(model="BAAI/bge-m3")
llm = ChatOpenAI(model="deepseek-ai/DeepSeek-V4-Flash")
vectorstore = PineconeVectorStore(
    index_name=os.environ['INDEX_NAME'],
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
    仅根据以下上下文回答问题:
    {context}
    问题: {question}
    提供一个详细的回答:
    """
)

def format_docs(docs):
    """将检索到的文档格式化为单个字符串"""
    return "\n\n".join(doc.page_content for doc in docs)
# ==========================
# 实现1：无LCEL(基于简单函数的方法)
# ==========================
def retrieval_chain_without_lcel(query: str):
    """
    无需LCEL的简单检索链。
    手动检索文档，格式化它们，并生成响应。

    限制条件:
    -手动逐步执行
    -没有内置流支持
    -没有额外代码的支持异步功能
    -更难与其他链组合
    -更冗长且易出错
    """
    # 步骤1：检索相关文档
    docs = retriever.invoke(query)

    # 步骤2：将文档格式化为上下文字符串
    context = format_docs(docs)

    # 步骤3：使用上下文和问题格式化消息
    messages = prompt_template.format_messages(context=context, question=query)

    # 步骤4：使用格式化后的消息列表（实际上只有一条消息）调用大语言模型
    response = llm.invoke(messages)

    # 步骤5：返回内容
    return response.content

# ==========================
# 实现方案2：使用LCEL(LangChainExpression语言)一一更优方法
# ==========================
def create_retrieval_chain_with_lcel():
    """
    使用LCEL(LangChain表达式语言)创建一个检索链。
    返回一个可以使用{"question": "..."}调用的链

    相较于非LCEL方法的优势:
    -声明式且可组合:易于使用管道操作符(|)链式执行操作
    -内置流处理:chain.stream()可直接使用
    -内置异步:提供chain.ainvoke()和chain.astream()函数
    -批处理:chain.batch()用于多个输入
    -类型安全:更好地集成LangChain的类型系统
    -更少代码:更简洁易读
    -可复用性:链可以保存、共享并与其他链组合
    -更好的调试:LangChain提供更好的可观测性工具
    """
    retrieval_chain = (
        retriever 
        | format_docs 
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain

if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "什么是智能体?"

    # 选项 0: 无RAG的原始调用
    # print("\n"+ "="* 70)
    # print("实现 0: 原始大模型调用 (No RAG)")
    # print("=" * 70)
    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print("\n答案:")
    # print(result_raw.content)

    # 选项1：使用不包含LCEL的实现
    print("\n"+ "="* 70)
    print("实现 1: 简单检索链 (Without LCEL)")
    print("=" * 70)
    result_simple = retrieval_chain_without_lcel(query)
    print("\n答案:")
    print(result_simple)