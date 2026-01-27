import os

import streamlit as st
import weave
from dotenv import load_dotenv
from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter


def app() -> None:
    load_dotenv(override=True)
    weave.init(os.getenv("WEAVE_PROJECT_NAME"))

    st.title("Indexing")

    clicked = st.button("実行")
    if not clicked:
        return

    # ロード
    loader = DirectoryLoader(
        # ../tmp/langchain-docs/src/langsmith/ ではないので注意
        path="./tmp/langchain-docs/src/langsmith/",
        glob="**/*.mdx",
        loader_cls=TextLoader,
    )
    raw_docs = loader.load()
    st.info(f"{len(raw_docs)} documents loaded.")

    # チャンク化
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(raw_docs)
    st.info(f"{len(docs)} documents chunked.")

    # インデクシング
    with st.spinner("Indexing documents..."):
        embeddings = init_embeddings(model="text-embedding-3-small", provider="openai")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./tmp/chroma",
        )
        vector_store.reset_collection()
        vector_store.add_documents(docs)
    st.success("Indexing completed.")


app()
