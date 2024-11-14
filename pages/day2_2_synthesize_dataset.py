import nest_asyncio
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client
from ragas.testset.evolutions import multi_context, reasoning, simple
from ragas.testset.generator import TestsetGenerator


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

    # Ragas用のmetadataを追加
    for document in documents:
        document.metadata["filename"] = document.metadata["source"]

    # 合成テストデータの生成
    nest_asyncio.apply()

    generator = TestsetGenerator.from_langchain(
        generator_llm=ChatOpenAI(model="gpt-4o-mini"),
        critic_llm=ChatOpenAI(model="gpt-4o-mini"),
        embeddings=OpenAIEmbeddings(),
    )

    with st.spinner("Generating testset..."):
        testset = generator.generate_with_langchain_docs(
            documents,
            test_size=4,
            distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
        )

    st.write(testset.to_pandas())

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

    for testset_record in testset.test_data:
        inputs.append(
            {
                "question": testset_record.question,
            }
        )
        outputs.append(
            {
                "contexts": testset_record.contexts,
                "ground_truth": testset_record.ground_truth,
            }
        )
        metadatas.append(
            {
                "source": testset_record.metadata[0]["source"],
                "evolution_type": testset_record.evolution_type,
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
