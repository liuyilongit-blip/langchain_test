import re
import inspect
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import ollama

# traceable(可追踪) 装饰器。它将接收我们的函数，并将其转换为可追踪的工具。这意味着我们可以在LangSmith中查看工具调用的详细信息。
from langsmith import traceable

# 限制循环次数
MAX_ITERATIONS= 10
MODEL = "qwen3:1.7b"  # 或其他模型

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

tools = {
    "get_product_price" : get_product_price,
    "apply_discount" : apply_discount
}

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # _wrapped_ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
        # 如果 tool_function 有 __wrapped__ 属性 → 返回 __wrapped__（原始未装饰的函数），如果没有 → 返回 tool_function 本身
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        # 获取函数的签名（参数和返回类型）
        signature = inspect.signature(original_function)
        # 获取函数的文档字符串，如果没有则返回空字符串
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)
tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
"STRICT RULES - you must follow these exactly:\n"
"1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price."
"2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price - do NoT pass a made-up number."
"3. NEVER calculate discounts yourself using math.Always use the apply_discount tool."
"4. If the user does not specify a discount tier,ask them which tier to use - do NOT assume one." 

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

# CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea it's an agent -
# all agency comes from the prompt above and our regex parsing below.
@traceable(name="ollama Chat",run_type="llm")
def ollama_chat_traced(model,messages,options):
    return ollama.chat(model=model, messages=messages, options=options)

# 将追踪函数内的所有操作，并归入同一个作用域。向代理循环添加代码时，所有操作都会嵌套在LangChain Agent Loop追踪记录下。
# 这对统计非常有用，如消耗了多少token。将所有内容嵌套在一个追踪记录下非常实用。
@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print(f"Question:{question}")
    print("="* 60)
    # CHANGE 5: One prompt string replaces the system/user message split.
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range (1, MAX_ITERATIONS):
        print(f"\n--- Iteration{iteration} ---")
        full_prompt = prompt + scratchpad

        # Stop token prevents the LLM from generating its own Observation -
        # we inject the real tool result instead.
        response = ollama_chat_traced(
            model = MODEL,
            messages = [{"role": "user","content": full_prompt}],
            options = {"stop": ["\nObservation"], "temperature": 0}
        )
        output = response.message.content
        print(f"LLM Output:\n{output}")

        print(f" [Parsing] Lookingfor Final Answer in LLMoutput...")
        # 正则表达式是现代所有代理和函数调用的起源
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f" [Parsed] Final Answer: {final_answer}")
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer


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
