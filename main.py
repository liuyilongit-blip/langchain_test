from typing import Any, Dict, List

import streamlit as st

from backend.core import run_llm


def _format_sources(context_docs: List[Any]) -> List[str]:
    """格式化并提取检索到的文档来源列表"""
    return [str(getattr(doc, "metadata", {}).get("source", "Unknown")) for doc in (context_docs or [])]
    # return [
    #     str((meta.get("source") or "Unknown"))
    #     for doc in (context_docs or [])
    #     if (meta := (getattr(doc, "metadata", None) or {})) is not None
    # ]


# 页面基础配置
st.set_page_config(page_title="LangChain 文档助手", layout="centered")
st.title("LangChain 文档助手")

# 侧边栏配置
with st.sidebar:
    st.subheader("会话")
    # 清空会话按钮
    if st.button("清除会话", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

# 初始化会话状态中的消息列表
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "您可以向我咨询任何关于 LangChain 文档的问题，我会检索相关内容并注明引用来源。",
            "sources": [],
        }
    ]

# 渲染历史聊天消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果有引用的来源，折叠展示
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# 获取用户输入的提问
prompt = st.chat_input("输入关于 LangChain 的问题…")
if prompt:
    # 记录并渲染用户发送的消息
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 助手生成回答
    with st.chat_message("assistant"):
        try:
            with st.spinner("检索文档并生成答案…"):
                # 调用后端 LLM 进行检索与回答
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(未返回答案。)"
                sources = _format_sources(result.get("context", []))

            # 展示回答内容及引用来源
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            # 将助手回答存入会话状态
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error("生成响应失败。")
            st.exception(e)


