import os
from dotenv import load_dotenv

load_dotenv()

os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
print('Ingesting...')
loader = TextLoader("./text.txt", encoding="utf-8")
document = loader.load()
print('splitter...')
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(document)
print('ingesting...')
embeddings = OpenAIEmbeddings(model="BAAI/bge-m3")
PineconeVectorStore.from_documents(texts, embeddings,index_name=os.environ['INDEX_NAME'])
print('finish')


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("111111111")
    