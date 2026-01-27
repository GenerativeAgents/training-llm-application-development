import os

import pandas as pd
import streamlit as st
import weave
from dotenv import load_dotenv
from weave import Dataset


def app() -> None:
    load_dotenv(override=True)
    weave.init(os.getenv("WEAVE_PROJECT_NAME"))

    st.title("Create Dataset")

    # data/examples.csvを読み込んで表示
    examples_df = pd.read_csv("data/examples.csv")
    st.write(examples_df)

    clicked = st.button("データセット作成")
    if not clicked:
        return

    # アップロード
    dataset = Dataset.from_pandas(examples_df)
    dataset.name = "training-llm-app"
    weave.publish(dataset)
    st.success("Dataset upload completed.")


app()
