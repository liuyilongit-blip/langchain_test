import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
# tool 装饰器。它将接收我们的函数，并转换为自定义工具。
from langchain.tools import tool
# ToolMessage是一个包含工具结果的消息。SystemMessage 是一个包装器，用于表示系统消息。HumanMessage 让我们可以使用单一接口来应对所有模型。
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
# 为Agent提供LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    temperature=0,
    model="gpt-5",  # 或其他模型
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_api_base="https://oa.api2d.net",
)

tools = []

# 限制循环次数
MAX_ITERATIONS= 10

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print (f"   >>Executing get_product_price(product='{product}')")
    prices = { "laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50 }
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print (f"   >>Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = { "bronze": 5, "silver": 12, "gold": 23 }
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

def run_agent(query: str):
    pass

def main():
    print("欢迎使用LangChain工具示例！")


if __name__ == "__main__":
    main()
