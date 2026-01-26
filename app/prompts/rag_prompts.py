"""
RAG関連のプロンプトテンプレート

weave.StringPromptを使用してプロンプトをバージョン管理します。
weave.publish()を呼び出すと、コンテンツハッシュベースの重複排除が行われるため、
同じ内容であれば何度publishしても新バージョンは作成されません。
"""

import weave

# 回答生成用プロンプト（naive, hyde, multi_query, rag_fusion, rerank, route, hybridで共通）
generate_answer_prompt = weave.StringPrompt(
    """以下の文脈だけを踏まえて質問に回答してください。

文脈: \"\"\"
{context}
\"\"\"

質問: {question}"""
)

# HyDE用の仮説的回答生成プロンプト
hypothetical_prompt = weave.StringPrompt(
    """次の質問に回答する一文を書いてください。

質問: {question}"""
)

# Multi-Query / RAG Fusion用のクエリ生成プロンプト
query_generation_prompt = weave.StringPrompt(
    """質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザーの質問に対して複数の視点を提供することが目標です。

質問: {question}"""
)

# Route用のルーティングプロンプト
route_prompt = weave.StringPrompt(
    """質問に回答するために適切なRetrieverを選択してください。
用意しているのは、LangSmithに関する情報を検索する「langsmith_document」と、
それ以外の質問をWebサイトで検索するための「web」です。

質問: {question}"""
)


def publish_rag_prompts() -> None:
    """
    RAGプロンプトをWeaveに公開します。

    Weaveはコンテンツハッシュベースの重複排除を行うため、
    同じ内容であれば何度呼び出しても新バージョンは作成されません。
    """
    weave.publish(generate_answer_prompt, name="generate_answer_prompt")
    weave.publish(hypothetical_prompt, name="hypothetical_prompt")
    weave.publish(query_generation_prompt, name="query_generation_prompt")
    weave.publish(route_prompt, name="route_prompt")
