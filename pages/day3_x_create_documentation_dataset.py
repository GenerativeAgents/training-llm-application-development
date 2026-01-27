import os

import pandas as pd
import streamlit as st
import weave
from dotenv import load_dotenv
from weave import Dataset


def app() -> None:
    load_dotenv(override=True)
    weave.init(os.getenv("WEAVE_PROJECT_NAME"))

    st.title("Create Documentation Agent Dataset")

    st.markdown("""
    このページでは、Documentation Agent 評価用のデータセットを Weave に登録します。

    **データセットのカラム:**
    - `user_request`: アプリケーション開発要求（Agentへの入力）
    - `expected_output`: 期待される出力の概要（LLM-as-judge用）
    - `key_requirements`: 含まれるべきキーワード（カンマ区切り）
    """)

    # data/documentation_agent_examples.csvを読み込んで表示
    csv_path = "data/documentation_agent_examples.csv"
    examples_df = pd.read_csv(csv_path)

    st.subheader("データセット内容")
    st.dataframe(examples_df, use_container_width=True)

    st.subheader("データセット統計")
    st.write(f"サンプル数: {len(examples_df)}")

    clicked = st.button("データセット作成", type="primary")
    if not clicked:
        return

    # アップロード
    dataset = Dataset.from_pandas(examples_df)
    dataset.name = "documentation-agent"
    weave.publish(dataset)
    st.success("Dataset upload completed. Weave UIで確認してください。")


app()
