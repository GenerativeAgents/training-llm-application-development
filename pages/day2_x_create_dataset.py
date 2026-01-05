import pandas as pd
import streamlit as st
import weave
from dotenv import load_dotenv
from weave import Dataset


def app() -> None:
    load_dotenv(override=True)
    weave.init("training-llm-app")

    st.title("Create Dataset")

    # data/examples.csvを読み込んで表示
    examples_df = pd.read_csv("data/examples.csv")
    st.write(examples_df)

    clicked = st.button("データセット作成")
    if not clicked:
        return

    # アップロード
    dataset = Dataset.from_pandas(examples_df)
    dataset.name = "dataset-example"
    weave.publish(dataset)
    st.success("Dataset upload completed.")


app()
