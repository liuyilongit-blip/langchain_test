import os

from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
# 从 LangChain 导入 tool 装饰器
from langchain.tools import tool
# 从 LangChain 核心模块导入 HumanMessage 类。使用 HumanMessage 来调用Agent，这将是Agent执行的输入。
from langchain_core.messages import HumanMessage
# 为Agent提供LLM
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Sourse(BaseModel):
    # 定义一个供 Agent 使用的来源数据结构（Schema）
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    # 包括答案和来源的代理响应数据结构
    """Schema for agent response with answer and sources"""

    # 描述为“代理对查询的回答”。代理的答案关于问题
    answer: str = Field(description="The agent's answer to the query")
    # 如果没有来源，则默认为空列表。列表关于来源，用于产生答案。
    sources: List[Sourse] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatOpenAI(
    temperature=0,
    model="gpt-5",  # 或其他模型
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_api_base="https://oa.api2d.net",
)

tools = [TavilySearch()]
# 这个代理实际上是一个 runnable（可运行对象）。response_format响应格式。
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    res = agent.invoke({"messages": HumanMessage(content="search for 3job postings for an ai engineer using langchain in the bay area on linkedin and list theirdetails")})
    print(res)  


if __name__ == "__main__":
    main()
