import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langsmith import Client


def app() -> None:
    load_dotenv(override=True)

    st.title("Create Dataset")

    # data/examples.csvを読み込んで表示
    examples_df = pd.read_csv("data/examples.csv")
    st.write(examples_df)

    clicked = st.button("データセット作成")
    if not clicked:
        return

    # LangSmithのDatasetの作成
    dataset_name = "training-llm-app"

    client = Client()

    if client.has_dataset(dataset_name=dataset_name):
        client.delete_dataset(dataset_name=dataset_name)

    dataset = client.create_dataset(dataset_name=dataset_name)

    # アップロードする形式に変換
    inputs = []
    outputs = []

    for _, example in examples_df.iterrows():
        inputs.append(
            {
                "question": example["question"],
            }
        )
        outputs.append(
            {
                "context": example["context"],
                "answer": example["answer"],
            }
        )

    # アップロード
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        dataset_id=dataset.id,
    )
    st.success("Dataset upload completed.")


app()
