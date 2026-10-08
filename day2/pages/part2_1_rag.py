import os
from pathlib import Path
from typing import TypedDict

import lancedb
import streamlit as st
import weave
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_feedback import streamlit_feedback
from weave.trace.weave_client import Call

from app.session_state import reset_session_state_on_page_change


class RagOutput(TypedDict):
    hits: list[dict[str, str]]
    answer: str | None


class Feedback(TypedDict):
    score: str
    text: str | None


DB_DIR = Path("data/lancedb")

load_dotenv(override=True)
client = OpenAI(max_retries=10)


@st.cache_resource
def init_weave() -> None:
    weave.init(os.environ["WANDB_PROJECT"])


# 文字列をベクトルに変換(埋め込み)
def embed(texts: list[str]) -> list[list[float]]:
    res = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in res.data]


# 質問文をベクトルにして、コサイン距離が近いチャンクの上位5件を取り出す
@weave.op
def search(question: str) -> list[dict[str, str]]:
    table = lancedb.connect(DB_DIR).open_table("simple_rag")
    hits = table.search(embed([question])[0]).distance_type("cosine").limit(5).to_list()
    return [{"source": hit["source"], "text": hit["text"]} for hit in hits]


# LLMに、取り出したチャンクを情報として与えて質問に答えさせる
@weave.op
def rag(question: str) -> RagOutput:

    # 検索の実行
    hits = search(question)

    # LLMの呼び出し
    documents = "\n\n".join(f"## {hit['source']}\n{hit['text']}" for hit in hits)
    res = client.chat.completions.create(
        model="gpt-6-luna",
        messages=[
            {
                "role": "system",
                "content": "与えられた文書だけを根拠に、質問に答えてください。",
            },
            {"role": "user", "content": f"# 文書\n{documents}\n\n# 質問\n{question}"},
        ],
    )
    return {"hits": hits, "answer": res.choices[0].message.content}


# フィードバックの送信
def send_feedback(feedback: Feedback, call: Call) -> None:
    """streamlit_feedback の送信内容(score は 👍 か 👎、text はコメント)を、このトレースに付ける。"""
    call.feedback.add_reaction(feedback["score"])
    if feedback["text"]:
        call.feedback.add_note(feedback["text"])
    st.success("ご意見ありがとうございました。")


def app() -> None:
    reset_session_state_on_page_change(__file__)
    init_weave()
    st.title("シンプルなRAG")

    question = st.text_input("質問")
    if st.button("実行") and question:
        # .call はトレース(call)も返す。フィードバックのボタンを押すと画面が再実行されるので、結果は session_state に取っておく
        st.session_state.output, st.session_state.call = rag.call(question)

    if "call" in st.session_state:
        output, call = st.session_state.output, st.session_state.call

        # 検索結果の表示
        st.subheader("検索結果(上位 5 件)")
        for hit in output["hits"]:
            with st.expander(hit["source"]):
                st.code(hit["text"], language=None, wrap_lines=True)

        # 回答の表示
        st.subheader("回答")
        st.markdown(output["answer"])

        # ユーザーフィードバックの受付
        # 👍 か 👎 を選ぶとコメント欄が出て、Submit を押すと send_feedback が呼ばれる
        streamlit_feedback(
            "thumbs",
            optional_text_label="コメント(任意)",
            align="flex-start",
            on_submit=send_feedback,
            args=(call,),
            key=f"feedback-{call.id}",
        )

        # Weaveのトレースのリンクを表示
        st.markdown(f"[Weave のトレース]({call.ui_url})")


app()
