from typing import Any

import streamlit as st
import weave

from app.advanced_rag.rag import DATA_DIR, PROJECT, PROMPTS, RagModel

CORPUS_DIR = DATA_DIR / "corpus"

# 段階 → RagModel の対応表
STAGES: dict[int, dict[str, Any]] = {
    1: {"index": "basic"},
    2: {"index": "basic", "use_collection": True},
    3: {"index": "structured", "use_collection": True},
    4: {"index": "vision", "use_collection": True},
    5: {"index": "vision", "use_collection": True, "search": "hybrid"},
    6: {"index": "vision", "use_collection": True, "search": "hybrid", "rerank": "llm"},
    7: {
        "index": "vision",
        "use_collection": True,
        "search": "hybrid",
        "rerank": "llm",
        **PROMPTS["collection"],
    },
    8: {
        "index": "vision",
        "use_collection": True,
        "search": "hybrid",
        "rerank": "llm",
        **PROMPTS["collection"],
        "select": "score",
        "expand": "unit+steps",
    },
}


class GuiRagModel(RagModel):
    def retrieve(self, question: str, collection: str | None = None) -> dict[str, Any]:  # type: ignore
        """検索が終わった時点で、LLM に渡すチャンクの一覧を出す。op にしないので、トレースは評価と同じ形になる。"""
        retrieved = super().retrieve(question, collection)
        st.subheader("LLM に渡す情報")
        corpus_dir = CORPUS_DIR.resolve()
        for i, c in enumerate(retrieved["contexts"]):
            with st.expander(c["id"]):
                st.caption(f"{c['title']}")
                st.code(c["text"], language=None, wrap_lines=True)
                path = (corpus_dir / c["file"]).resolve()
                if not path.is_relative_to(corpus_dir) or not path.is_file():
                    st.caption("元のファイルが見つからないため、ダウンロードできません。")
                    continue
                st.download_button(
                    "元のファイルをダウンロード",
                    path.read_bytes,
                    file_name=path.name,
                    on_click="ignore",
                    key=f"source-{i}-{c['id']}",
                )
        return retrieved


@st.cache_resource
def init_weave() -> None:
    weave.init(PROJECT)


def app() -> None:
    init_weave()
    st.title("Advanced RAG")

    with st.sidebar:
        stage = st.selectbox(
            "試すロジック", list(STAGES), format_func=lambda n: f"段階 {n}"
        )
        attrs = STAGES[stage]
        collection = None
        if attrs.get("use_collection"):
            collection = st.selectbox(
                "検索対象", sorted(p.name for p in CORPUS_DIR.iterdir() if p.is_dir())
            )

    question = st.text_input("質問")
    # disabled=not question にすると、入力欄からフォーカスが外れるまで押せない(値はそのときに送られる)
    if st.button("実行") and question:
        model = GuiRagModel(**attrs)
        # .call はトレース(call)も返す。self を明示して渡す。例外は握りつぶさずに画面に出す
        output, call = model.predict.call(
            model, question, collection, __should_raise=True
        )
        st.subheader("回答")
        st.markdown(output["answer"])
        st.markdown(f"[Weave のトレース]({call.ui_url})")


app()
