import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

# 配置 SSL 上下文以使用 certifi 证书
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    show_progress_bar=False, # 显示进度条
    chunk_size=50, # 每个批次的最大文档数
    retry_min_seconds=10, # 重试间隔参数
)
vectorstore = PineconeVectorStore(index_name="langchain-doc-index",embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=2, max_breadth=10, max_pages=500)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """异步批量处理文档。"""
    log_header("向量存储阶段")
    log_info(
        f"📚 向量库索引: 准备将 {len(documents)} 篇文档添加到向量数据库",
        Colors.DARKCYAN,
    )

    # 创建分批数据
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 向量库索引: 已拆分为 {len(batches)} 个批次，每批 {batch_size} 篇文档"
    )

    # 并发处理所有批次
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"向量库索引: 成功添加批次 {batch_num}/{len(batches)}（包含 {len(batch)} 篇文档）"
            )
        except Exception as e:
            log_error(f"向量库索引: 添加批次 {batch_num} 失败 - {e}")
            return False
        return True

    # 并发执行批次任务
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 统计成功批次数量
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"向量库索引: 所有批次均处理成功！（{successful}/{len(batches)}）"
        )
    else:
        log_warning(
            f"向量库索引: 成功处理 {successful}/{len(batches)} 个批次"
        )


async def main():
    """协调整个流程的主异步函数。"""
    log_header("文档摄取流水线")

    log_info(
        "🗺️  TavilyCrawl: 开始爬取文档网站",
        Colors.PURPLE,
    )
    # 爬取文档网站

    res = tavily_crawl.invoke(
        {
            "url": "https://docs.langchain.com/oss/python/",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    # 将 Tavily 爬取结果转换为 LangChain Document 对象
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: 成功从文档网站爬取 {tavily_crawl_result_item['url']}"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # 将文档切分为文本块 (Chunk)
    log_header("文档分块阶段")
    log_info(
        f"✂️  文本分割器: 正在处理 {len(all_docs)} 篇文档（分块大小 4000，重叠 200）",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"文本分割器: 从 {len(all_docs)} 篇文档中成功创建了 {len(splitted_docs)} 个分块"
    )

    # 异步处理文档入库
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("流水线执行完毕")
    log_success("🎉 文档摄取流水线成功完成！")
    log_info("📊 总结统计:", Colors.BOLD)
    log_info(f"   • 提取文档数: {len(all_docs)}")
    log_info(f"   • 创建分块数: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
