import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def invoke_llm(messages: list[BaseMessage]) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder("messages", optional=True),
        ]
    )
    model = init_chat_model(
        model="gpt-4.1-nano", model_provider="openai", temperature=0
    )
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"messages": messages})


def app() -> None:
    load_dotenv(override=True)

    st.title("Simple Chatbot")

    # 会話履歴を初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    messages: list[BaseMessage] = st.session_state.messages

    # 会話履歴を表示
    for message in messages:
        with st.chat_message(message.type):
            st.write(message.content)

    # ユーザーの入力を受け付ける
    human_message = st.chat_input()
    if not human_message:
        return

    # ユーザーの入力を表示
    with st.chat_message("human"):
        st.write(human_message)

    # ユーザーの入力を会話履歴に追加
    messages.append(HumanMessage(content=human_message))

    # 応答を生成して表示
    ai_message = invoke_llm(messages)
    with st.chat_message("ai"):
        st.write(ai_message)

    # LLMの応答を会話履歴を追加
    messages.append(AIMessage(content=ai_message))


app()
