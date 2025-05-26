import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.documentation_agent.agent import DocumentationAgent, InterviewState


def app() -> None:
    load_dotenv(override=True)

    with st.sidebar:
        model_name = st.selectbox(
            label="モデル", options=["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
        )
        temperature = st.slider(
            label="temperature", min_value=0.0, max_value=1.0, value=0.0
        )
        persona_count = st.number_input(
            label="生成するペルソナの人数",
            min_value=1,
            max_value=10,
            value=5,
        )

    st.title("Documentation Agent")

    user_request = st.text_input(
        label="作成したいアプリケーションについて記載してください",
        value="スマートフォン向けの健康管理アプリを開発したい",
    )

    clicked = st.button("実行", disabled=not user_request)
    if not clicked:
        return

    # ChatOpenAIモデルを初期化
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    agent = DocumentationAgent(llm=llm, k=persona_count)
    # エージェントを実行して最終的な出力を取得
    initial_state = InterviewState(user_request=user_request)

    with st.spinner("実行中..."):
        for chunk in agent.graph.stream(initial_state, stream_mode="updates"):
            keys = chunk.keys()
            for key in keys:
                state = chunk[key]

                if key == "generate_personas":
                    with st.expander("生成されたペルソナ"):
                        for persona in state["personas"]:
                            st.write(f"## {persona.name}\n\n{persona.background}")
                elif key == "conduct_interviews":
                    with st.expander("インタビュー結果"):
                        for i, interview in enumerate(state["interviews"]):
                            st.write(
                                f"## インタビュー結果 {i + 1}\n\n"
                                f"### ペルソナ\n\n{interview.persona.name}\n\n"
                                f"### 質問\n\n{interview.question}\n\n"
                                f"### 回答\n\n{interview.answer}"
                            )
                elif key == "evaluate_information":
                    with st.expander("情報の評価"):
                        is_information_sufficient = state["is_information_sufficient"]
                        st.write(
                            f"is_information_sufficient: {is_information_sufficient}"
                        )
                elif key == "generate_requirements":
                    st.write(state["requirements_doc"])


app()
