import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from langchain_agents import create_agent
# 从 LangChain 导入 tool 装饰器
from langchain_core.tools import tool
# 从 LangChain 核心模块导入 HumanMessage 类。使用 HumanMessage 来调用Agent，这将是Agent执行的输入。
from langchain_core import HumanMessage
# 为Agent提供LLM
from langchain_openai import ChatOpenAI



load_dotenv()

def main():
    iformation = """
    埃隆·里夫·马斯克（Elon Reeve Musk），1971年6月28日出生于南非行政首都比勒陀利亚，美国、南非、加拿大三重国籍的企业家、工程师、发明家、慈善家，特斯拉创始人兼首席执行官 [39] [78]，SpaceX首席执行官兼首席技术官，SolarCity董事会主席 [1]、Twitter首席执行官 [95]，Neuralink创始人 [99]、OpenAI联合创始人 [96]，美国国家工程院院士 [49]，英国皇家学会院士 [23]，美国党创始人 [591]。
马斯克本科毕业于宾夕法尼亚大学经济学和物理学双专业，从小就对科学技术十分痴迷，10岁开始学习编程，13岁开发出一款游戏并因此赚到人生第一桶金。1995年至2002年，与合伙人创办Zip2和PayPal [3]。2002年，投资成立SpaceX [10]。2004年，向Tesla公司投资630万美元 [4]。2006年，投资创办光伏发电企业SolarCity [2]。 [57-58]马斯克在汽车、能源、航空航天以及人工智能等领域均取得了诸多成就。
马斯克曾多次入选《时代周刊》全球最具影响力人物 [45] [65] [144]，获得2019年霍金科学传播奖 [103]，2023年度世界航天奖 [104]。2021年胡润发布榜单，马斯克以1.28万亿首次成为世界首富 [33]；10月，其财富突破3000亿美元，成为福布斯统计史上最富有的人 [40]。 [59] [82] [90] [110] [119] [141-142] [153]2024年首破4000亿美元。 [311-312]2026年2月，成为史上首位身价突破8000亿美元的富豪，达到创纪录的8520亿美元； [695] [700] [732]3月，美议员提议对亿万富翁征收5%年度财富税，马斯克首年或缴420亿美元。
2025年5月30日，马斯克表示以后将作为总统顾问访问政府效率部。 [523] [526]12月，入选《时代》周刊2025“年度人物”。 [691-692]2026年3月5日，以55400亿元人民币位列《2026胡润全球富豪榜》第1位 [741]。
    """
    system_prompt = """
    根据关于一个人的信息{information}，我想要你创建:
    1.一个简短的总结
    2.关于他们的两个有趣事实
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=system_prompt,
    )

    llm = ChatOpenAI(
        temperature=0,
        model="Qwen/Qwen2.5-7B-Instruct",  # 或其他模型
        openai_api_key=os.environ.get("SILICONFLOW_API_KEY"),
        openai_api_base="https://api.siliconflow.cn/v1",
    )
    chain = summary_prompt_template | llm
    result = chain.invoke({"information": iformation})
    print(result.content)

if __name__ == "__main__":
    main()
