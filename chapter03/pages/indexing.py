import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


def app() -> None:
    load_dotenv(override=True)

    st.title("Indexing")

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
    raw_docs = loader.load()
    st.info(f"{len(raw_docs)} documents loaded.")

    # チャンク化
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(raw_docs)
    st.info(f"{len(docs)} documents chunked.")

    # インデクシング
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    with st.spinner("Indexing documents..."):
        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./tmp/chroma",
        )
    st.success("Indexing completed.")


app()
