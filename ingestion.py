import asyncio
import os
# import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

def chunk_urls(urls: List[str], chunk_size: int = 3) -> List[List[str]]:
    """将 URL 分割成指定大小的块"""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """从一批 URL 中提取文档"""
    try:
        log_info(
            f"🔄 正在处理第 {batch_num} 批次，包含 {len(urls)} 个 URL",
            Colors.BLUE,
        )
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(
            f"TavilyExtract：已完成第 {batch_num} 批次 - 提取了 {len(docs.get('results',[]))} 份文档"
        )
        return docs
    except Exception as e:
        log_error(f"TavilyExtract：提取第 {batch_num} 批次失败 - {e}")
        return []

async def async_extract(url_batches: List[List[str]]):
    log_header("文档提取阶段")
    log_info(
        f"TavilyExtract：开始并发提取 {len(url_batches)} 个批次",
        Colors.DARKCYAN
    )
    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]
    # return_exceptions=True 的作用是：当某个 task 抛出异常时，不会中断整个 gather()，而是将异常作为结果返回到列表中。
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # 过滤掉异常，并将结果扁平化。
    all_pages = []
    failed_batches = 0 # 记录失败的批次数量
    for result in results:
        # 判断 result 是否是一个异常对象。isinstance(...)：判断对象是否属于某个类型。
        if isinstance(result, Exception): # Exception：Python 中所有常见异常的基类
            log_error(f"TavilyExtract：批次处理失败，出现异常 - {result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]: # 类型:忽略
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source":extracted_page["url"]},
                )
                all_pages.append(document)
    log_success(
        f"TavilyExtract：提取完成！共提取页面数: {len(all_pages)}"
    )
    if failed_batches > 0:
        log_warning(f"TavilyExtract：{failed_batches} 个批次在提取过程中失败")
    return all_pages

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
        "max_depth":3,
        "extract_depth":"advanced",
    })
    log_success(
        f"TavilyMap：已成功从文档站点映射了 {len(site_map['results'])} 个 URL"
    )

    # 将 URL 分成每批 20 个
    url_batches = chunk_urls(list(site_map['results']), chunk_size=20)
    log_info(
        f"URL 处理：将 {len(site_map['results'])} 个 URL 拆分为 {len(url_batches)} 个批次",
        Colors.BLUE,
    )

    # 从 URL 中提取文档
    all_docs = await async_extract(url_batches)

    # 将文档切分成多个文本块
    log_header("文档切分阶段")
    log_info(
        f"文本切分器：正在将 {len(all_docs)} 个文档按每块 4000 字符进行切分，相邻文本块重叠 200 字符。",
        Colors.YELLOW,
    )
    # RecursiveCharacterTextSplitter是langchain的文本切分器，它会根据字符数将文档切分为多个文本块，并且可以设置相邻文本块的重叠字符数。
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"文本切分器：已将 {len(all_docs)} 个文档切分为 {len(splitted_docs)} 个文本块。"
    )

if __name__ == "__main__":
    asyncio.run(main())
    