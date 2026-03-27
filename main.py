import os

from dotenv import load_dotenv

from langchain.agents import create_agent
# 从 LangChain 导入 tool 装饰器
from langchain.tools import tool
# 从 LangChain 核心模块导入 HumanMessage 类。使用 HumanMessage 来调用Agent，这将是Agent执行的输入。
from langchain_core.messages import HumanMessage
# 为Agent提供LLM
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()

load_dotenv()
# 初始化一个 ChatOpenAI 实例
llm = ChatOpenAI(
    temperature=0,
    model="Qwen/Qwen2.5-7B-Instruct",  # 或其他模型
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_api_base="https://api.siliconflow.cn/v1",
)

@tool
def search(query: str) -> str:
    # Google 风格（最流行）：自动生成文档，可读性好且被 VS Code 等工具原生支持
    """模拟一个天气查询工具，返回天气信息。

    Args:
        query (str): 城市名称

    Returns:
        str: 天气信息字符串
    """
    
    return tavily.search(query=query)

tools = [search]
# 这个代理实际上是一个 runnable（可运行对象）
agent = create_agent(model=llm, tools=tools)

def main():
    res = agent.invoke({"messages": HumanMessage(content="请查询一下：北京的天气")})
    print(res)


if __name__ == "__main__":
    main()
