import time

import nest_asyncio
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator


def app() -> None:
    st.title("Synthesize Dataset")

    clicked = st.button("実行")
    if not clicked:
        return

    # ロード
    loader = DirectoryLoader(
        # ../tmp/langchain ではないので注意
        path="tmp/langchain",
        glob="**/*.mdx",
        loader_cls=TextLoader,
    )
    documents = loader.load()
    st.info(f"{len(documents)} documents loaded.")

    # 合成テストデータの生成
    nest_asyncio.apply()

    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    generator = TestsetGenerator(llm=llm)

    with st.spinner("Generating testset..."):
        start_time = time.time()

        testset = generator.generate_with_langchain_docs(
            documents,
            testset_size=4,
            transforms_embedding_model=embeddings,
        )

        end_time = time.time()

    elapsed_time = end_time - start_time
    st.success(f"Testset generated. Elapsed time: {elapsed_time:.2f} sec.")

    st.write(testset.to_pandas())

    st.write(testset.to_list())

    # LangSmithのDatasetの作成
    dataset_name = "training-llm-app"

    client = Client()

    if client.has_dataset(dataset_name=dataset_name):
        client.delete_dataset(dataset_name=dataset_name)

    dataset = client.create_dataset(dataset_name=dataset_name)

    # アップロードする形式に変換
    inputs = []
    outputs = []
    metadatas = []

    for testset_record in testset.to_list():
        inputs.append(
            {
                "question": testset_record["user_input"],
            }
        )
        outputs.append(
            {
                "contexts": testset_record["reference_contexts"],
                "ground_truth": testset_record["reference"],
            }
        )
        metadatas.append(
            {
                # "source": testset_record.metadata[0]["source"],
                "synthesizer_name": testset_record["synthesizer_name"],
            }
        )

    # アップロード
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        metadata=metadatas,
        dataset_id=dataset.id,
    )
    st.success("Dataset upload completed.")


app()
