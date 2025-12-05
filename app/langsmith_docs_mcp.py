from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="langsmith-docs-mcp")


@mcp.tool()
def langsmith_docs_retriever(query: str) -> list[Document]:
    """LangSmithの最新ドキュメントを検索します。LangSmithについて質問された場合に使用してください。"""

    load_dotenv(override=True)

    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory="./tmp/chroma",
    )
    retriever = vector_store.as_retriever()
    return retriever.invoke(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
