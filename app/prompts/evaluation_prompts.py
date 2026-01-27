"""
評価用のプロンプトテンプレート

weave.StringPromptを使用してプロンプトをバージョン管理します。
"""

import weave

# ハルシネーション評価用プロンプト（LangSmith Online Evaluatorの日本語訳）
answer_hallucination_prompt = weave.StringPrompt(
    """あなたは、モデル出力の幻覚（ハルシネーション）を評価する専門的なデータラベラーです。以下のルーブリックに基づいてスコアを割り当てることがあなたのタスクです：

<Rubric>
  幻覚のない回答とは：
  - 入力コンテキストによって直接裏付けられる検証可能な事実のみを含む
  - 裏付けのない主張や仮定を行わない
  - 推測的または想像上の詳細を追加しない
  - 日付、数字、および具体的な詳細において完全な正確性を保つ
  - 情報が不完全な場合は適切に不確実性を示す
</Rubric>

<Instructions>
  - 入力コンテキストを徹底的に読む
  - 出力で行われているすべての主張を特定する
  - 各主張を入力コンテキストと相互参照する
  - 裏付けのない情報や矛盾する情報を記録する
  - 幻覚の重大性と数量を考慮する
</Instructions>

<Reminder>
  事実の正確性と入力コンテキストからの裏付けのみに焦点を当ててください。採点においてスタイル、文法、またはプレゼンテーションは考慮しないでください。短くても事実に基づく回答は、裏付けのない主張を含む長い回答よりも高いスコアを付けるべきです。
</Reminder>

出力の幻覚を評価するために、以下のコンテキストを使用してください：
<context>
{context}
</context>

<input>
{input}
</input>

<output>
{output}
</output>

利用可能な場合は、以下の参照出力も回答の幻覚を特定するのに役立てることができます：
<reference_outputs>
{reference_outputs}
</reference_outputs>"""
)


def publish_evaluation_prompts() -> None:
    """
    評価プロンプトをWeaveに公開します。

    Weaveはコンテンツハッシュベースの重複排除を行うため、
    同じ内容であれば何度呼び出しても新バージョンは作成されません。
    """
    weave.publish(answer_hallucination_prompt, name="answer_hallucination_prompt")
