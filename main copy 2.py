import os
from dotenv import load_dotenv

load_dotenv()

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import ollama

# traceable(可追踪) 装饰器。它将接收我们的函数，并将其转换为可追踪的工具。这意味着我们可以在LangSmith中查看工具调用的详细信息。
from langsmith import traceable

# 限制循环次数
MAX_ITERATIONS= 10
MODEL = "qwen3:4b"  # 或其他模型

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print (f"   >>Executing get_product_price(product='{product}')")
    prices = { "laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50 }
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print (f"   >>Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = { "bronze": 5, "silver": 12, "gold": 23 }
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# Difference 2: Without @tool, we must MANUALLY define the JSoN schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            # 这类似于函数的描述
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'"
                    }
                },
                "required": ["product"] # 这个参数(product)是必需的
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price"
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount.tier:'bronze','silver',or 'gold'"
                    }
                },
                "required": ["price", "discount_tier"] # 这两个参数(price, discount_tier)都是必需的
            }
        }
    }
]

# NOTE:Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to-LangChain's.@tool decorator):
# tools_for_llm.=.[get_product_price,apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section, For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSoN version here so you can see what @tool hides from you.

# --- Helper:traced 0llama call ---
# Difference 3:Without LangChain, we must manually trace LLM calls for LangSmith.

@traceable(name="ollama Chat",run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)

# 将追踪函数内的所有操作，并归入同一个作用域。向代理循环添加代码时，所有操作都会嵌套在LangChain Agent Loop追踪记录下。
# 这对统计非常有用，如消耗了多少token。将所有内容嵌套在一个追踪记录下非常实用。
@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    tools_dict= {
        "get_product_price" : get_product_price,
        "apply_discount" : apply_discount
    }
    messages = [
        {
            "role": "system",
            "content": (
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
        },
        { "role": "user","content": question }
    ]
    for iteration in range (1, MAX_ITERATIONS):
        print(f"\n--- Iteration{iteration} ---")
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message
        tool_calls = ai_message.tool_calls
        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        # Process only the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        # 方便追踪
        tool_call_id = tool_call.get("id")
        print(f"[Tool Selected]{tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Difference 7: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)
        print(f"[ToolResult]{observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role":"tool",
                "content":str(observation),
            }
        )
    print("ERROR: Max iterations reached without a final answer")
    return None

if __name__ == "__main__":
    result = run_agent("what is the price of a laptop after applying a gold discount?")
