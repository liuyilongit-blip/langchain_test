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
# traceable(可追踪) 装饰器。它将接收我们的函数，并将其转换为可追踪的工具。这意味着我们可以在LangSmith中查看工具调用的详细信息。
from langsmith import traceable

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

# 将追踪函数内的所有操作，并归入同一个作用域。向代理循环添加代码时，所有操作都会嵌套在LangChain Agent Loop追踪记录下。
# 这对统计非常有用，如消耗了多少token。将所有内容嵌套在一个追踪记录下非常实用。
# @traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tool_dict = {t.name: t for t in tools}
    llm = ChatOpenAI(
        temperature=0,
        model="Qwen/Qwen2.5-7B-Instruct",  # 或其他模型
        openai_api_key=os.environ.get("siliconflow_API_KEY"),
        openai_api_base="https://api.siliconflow.cn/v1"
    )
    # llm = init_chat_model(f"ollama:{MODEL}",temperature=0)
    # 接收工具列表，并将工具绑定到模型上。bind_tools只适用于支持函数调用的 LLM
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant."
                "You have access to a product catalog tool"
                "and a discount tool.\n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price."
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price - do NoT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math."
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier,"
                "ask them which tier to use - do NOT assume one."                
            )
        ),
        HumanMessage(content=question)
    ]
    for iteration in range (1, MAX_ITERATIONS):
        print(f"\n--- Iteration{iteration} ---")
        ai_message = llm_with_tools.invoke(messages)

    return messages

if __name__ == "__main__":
    result = run_agent("what is the price of a laptop after applying a gold discount?")
    print("123")
