import asyncio
import os
# import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chromadb import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

# Configure SSL context to use certifi certificates
# ssl_context = ssl.create_default_context(cafile=certifi.where())
# os.environ["SSL_CERT_FILE"] = certifi.where()
# os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    show_progress_bar=False, # 显示进度条
    chunk_size=50, # 每个批次的最大文档数
    retry_min_seconds=10, # 重试间隔参数
)
# chroma=Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(index_name="langchain-doc-index",embedding=embeddings)

# from langchain_community.document_loaders import TextLoader
# print('Ingesting...')
# loader = TextLoader("./text.txt", encoding="utf-8")
# document = loader.load()
# print('splitter...')
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(document)
# print('ingesting...')
# PineconeVectorStore.from_documents(texts, embeddings,index_name=os.environ['INDEX_NAME'])
# print('finish')

tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

async def main(): # 主协程
    """用于编排整个流程的主异步函数"""
    log_header("文档摄入管道")
    log_info(
        "TavilyMap：开始从以下位置映射文档结构 https://docs.langchain.com/oss/python/",
        Colors.PURPLE,
    )
    # 映射文档结构
    site_map = tavily_map.invoke({
        "url":"https://docs.langchain.com/oss/python/",
        "max_depth":1,
        "extract_depth":"advanced",
        "instructions":"请获取有关智能体的内容"
    })
    log_success(
        f"TavilyMap：已成功从文档站点映射了 {len(site_map['results'])} 个 URL"
    )

if __name__ == "__main__":
    asyncio.run(main())
    