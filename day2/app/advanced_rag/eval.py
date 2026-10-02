"""RagModel を Weave の Evaluation で評価する。

Weave に登録済みのデータセット(notebooks/part3_1_eval.ipynb で登録)の各質問について、次の 2 つを採点する。

- retrieval_hit: 正解の文書(source_ids)が検索結果に入っているか。source_ids は「答えるのに要る文書の組」のリストで、
  別の文書の組からも答えられる場合は組を複数並べる(例: [[テーブル定義書, ドメイン定義書], [メッセージ設計書]])。
  hit はどれか 1 組が全部入っていれば true、recall は組ごとの入っていた割合の最大値。
  source_id は `相対パス#シート名` でよく、そのシートのどのチャンク(#1, #2, ...)が
  当たっても正解とする(抽出器によって分割数が変わるため)。PDF の正解はページ(`#page18`)で、
  ページをまたぐチャンク(`#page18-21`)はその範囲のページの正解として数える。
  診断用に、正解の各文書が検索結果の何位に出たか(ranks、rank_depth 位より下は None)と、
  どれか 1 組を揃えるのに要る件数(needed_k、組の中の最下位の順位の、組の間での最小値)も記録する
- answer_correct: 回答が expected と同じ内容か(LLM による判定)

    uv run python -m app.advanced_rag.eval --dataset qa-advance            # 最新バージョン
    uv run python -m app.advanced_rag.eval --dataset qa-advance:<digest>   # バージョンを固定
    uv run python -m app.advanced_rag.eval --top-k 4                     # RagModel の属性を変えて測る
    uv run python -m app.advanced_rag.eval --use-collection              # 検索対象を各行の collection(資料群)に限定する
    uv run python -m app.advanced_rag.eval --trials 3                    # 各質問を 3 回ずつ解かせる(回答のぶれをならす)
    uv run python -m app.advanced_rag.eval --search hybrid               # ベクトル検索と全文検索を RRF で混ぜる(--rrf-k、--fts-query、--fts-fields)
    uv run python -m app.advanced_rag.eval --search hybrid --rerank llm  # 検索の上位 20 件を LLM が採点し直して並べ替える(--rerank-depth)
    uv run python -m app.advanced_rag.eval --prompt rules                # 回答用の system prompt を替える(base / rules / evidence / collection。rag.PROMPTS)
    uv run python -m app.advanced_rag.eval --reasoning-effort medium     # 回答(とキーワード・並べ直し)の reasoning effort を替える
    uv run python -m app.advanced_rag.eval --rerank llm --select score   # 並べ直しの点が閾値以上のチャンクだけを渡す(--score-threshold など)
    uv run python -m app.advanced_rag.eval --expand unit+steps           # 同じ区切りのほかのチャンクと、手順の途中から始まる区切りの手前のチャンクも渡す
    uv run python -m app.advanced_rag.eval --label 段階4                 # Weave の一覧に出る評価の名前(研修の段階を見分ける)
    uv run python -m app.advanced_rag.eval --index vision                # 検索するインデックス(作ったときの抽出器)。無ければ作るか確認する
"""

import argparse
import asyncio
import json
import os
import re
import sys

import weave
from pydantic import BaseModel, Field

from . import build_index
from .extractors import EXTRACTORS
from .rag import PROJECT, PROMPTS, RagModel, client, has_index, index_meta

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-6-luna")
JUDGE_REASONING_EFFORT = os.environ.get("JUDGE_REASONING_EFFORT", "low")  # 推論モデルでないなら空にする

JUDGE_PROMPT = (
    "質問に対する回答を採点します。回答が正解と同じ内容を含んでいれば correct を true に、"
    "誤りや欠落があれば false にしてください。表現の違いは問いません。"
)


class Judgement(BaseModel):
    """判定の形。Structured Outputs で、必ずこの形で返ってくる。"""
    correct: bool
    reason: str = Field(description="理由を一文で")


PAGE_RE = re.compile(r"page(\d+)")
PAGE_RANGE_RE = re.compile(r"page(\d+)-(\d+)(?:#|$)")


def covers(doc_id: str, source_id: str) -> bool:
    """検索結果のチャンク doc_id が、正解の source_id に当たるか。

    `相対パス#シート名` は、そのシートのどのチャンク(`#1`, `#2`, ...)でも当たり。
    `相対パス#page18` は、ページ範囲のチャンク(`#page18-21`、`#page15-18#2` など)が 18 を含めば当たり。
    """
    if doc_id == source_id or doc_id.startswith(source_id + "#"):
        return True
    path, _, unit = source_id.rpartition("#")
    page = PAGE_RE.fullmatch(unit)
    if not page or not doc_id.startswith(path + "#"):
        return False
    span = PAGE_RANGE_RE.match(doc_id, len(path) + 1)
    return bool(span) and int(span[1]) <= int(page[1]) <= int(span[2])


@weave.op
def retrieval_hit(source_ids: list[list[str]], output: dict) -> dict:
    ids = [c["id"] for c in output["contexts"]]
    recalls = []
    for group in source_ids:  # どれか 1 組が揃えば当たり
        found = [any(covers(i, s) for i in ids) for s in group]
        recalls.append(sum(found) / len(found))
    ranks = [[rank_of(s, output["ranked_ids"]) for s in group] for group in source_ids]
    needs = [max(r) for r in ranks if None not in r]
    return {"hit": max(recalls) == 1, "recall": max(recalls), "ranks": ranks, "needed_k": min(needs, default=None)}


def rank_of(source_id: str, ranked_ids: list[str]) -> int | None:
    """source_id に当たるチャンクの最上位の順位(1 始まり)。ranked_ids に無ければ None。"""
    return next((n for n, i in enumerate(ranked_ids, 1) if covers(i, source_id)), None)


@weave.op
def answer_correct(question: str, expected: str, output: dict) -> dict:
    res = client.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": f"# 質問\n{question}\n\n# 正解\n{expected}\n\n# 回答\n{output['answer']}",
            },
        ],
        response_format=Judgement,
        **({"reasoning_effort": JUDGE_REASONING_EFFORT} if JUDGE_REASONING_EFFORT else {}),
    )
    return res.choices[0].message.parsed.model_dump()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="qa-advance", help="Weave 上のデータセット名。name:version でバージョンを固定できる")
    parser.add_argument("--index", choices=list(EXTRACTORS), default=RagModel.model_fields["index"].default,
                        help="検索するインデックス(作ったときの抽出器)。無ければ作るかを確認する")
    parser.add_argument("--top-k", type=int, default=RagModel.model_fields["top_k"].default)
    parser.add_argument("--use-collection", action="store_true", help="検索対象を各行の collection 列の資料群に限定する")
    parser.add_argument("--trials", type=int, default=1, help="各質問を解かせる回数。回答は同じ条件でもぶれるので、比べるときは複数回にする")
    parser.add_argument("--search", choices=["vector", "fts", "hybrid"], default=RagModel.model_fields["search"].default)
    parser.add_argument("--fts-query", choices=["keywords", "pos", "raw"], default=RagModel.model_fields["fts_query"].default,
                        help="全文検索に渡す語。pos は形態素の品詞で抜き出したキーワード、keywords は LLM が抜き出したキーワード、raw は質問文そのまま")
    parser.add_argument("--fts-fields", nargs="+", choices=["morph", "bigram"],
                        default=RagModel.model_fields["fts_fields"].default, help="全文検索で見る列(形態素 / bi-gram)")
    parser.add_argument("--rrf-k", type=int, default=RagModel.model_fields["rrf_k"].default)
    parser.add_argument("--rerank", choices=["none", "llm"], default=RagModel.model_fields["rerank"].default,
                        help="llm なら検索の上位 --rerank-depth 件を LLM が採点し直して並べ替える")
    parser.add_argument("--rerank-depth", type=int, default=RagModel.model_fields["rerank_depth"].default)
    parser.add_argument("--prompt", choices=list(PROMPTS), default="base",
                        help="回答用の system prompt。rules は省略・脚色を禁じる規則を足し、evidence はさらに根拠の行を先に抜き書きさせる。"
                             "collection は rules に資料群ごとの説明を足す(--use-collection のときだけ)")
    parser.add_argument("--reasoning-effort", default=RagModel.model_fields["reasoning_effort"].default)
    parser.add_argument("--select", choices=["top_k", "score"], default=RagModel.model_fields["select"].default,
                        help="score なら並べ直しの点が --score-threshold 以上のチャンクだけを渡す(--min-selected〜--max-selected 件。--rerank llm のとき)")
    parser.add_argument("--score-threshold", type=int, default=RagModel.model_fields["score_threshold"].default)
    parser.add_argument("--min-selected", type=int, default=RagModel.model_fields["min_selected"].default)
    parser.add_argument("--max-selected", type=int, default=RagModel.model_fields["max_selected"].default)
    parser.add_argument("--expand", choices=["none", "unit", "unit+steps"], default=RagModel.model_fields["expand"].default,
                        help="unit なら同じ区切りのほかのチャンクも渡す。unit+steps なら、区切りが手順の途中から始まっていれば手前のチャンクも"
                             "(--max-back、--max-context-chars)")
    parser.add_argument("--max-back", type=int, default=RagModel.model_fields["max_back"].default)
    parser.add_argument("--label", help="Weave の一覧に出る評価の名前(省略時は Weave が付ける)")
    parser.add_argument("--max-context-chars", type=int, default=RagModel.model_fields["max_context_chars"].default)
    args = parser.parse_args()

    # インデックスは事前に配置しておく想定。無いときだけ、コストを示して作るかを確認する
    if not has_index(args.index):
        cost = ("Vision LLM を約 520 回呼ぶ。約 20 分、$1 未満。data/cache/ に説明文のキャッシュがあれば 2〜3 分"
                if args.index == "vision" else "1〜2 分")
        if input(f"インデックス {args.index} がありません。作りますか?({cost}) [y/N] ").strip().lower() != "y":
            sys.exit(f"uv run python -m app.advanced_rag.build_index --extractor {args.index} で作ってから実行してください")
        build_index.build(args.index, RagModel.model_fields["embedding_model"].default)

    weave.init(PROJECT)
    try:
        dataset = weave.ref(args.dataset).get()
    except Exception as e:
        sys.exit(f"データセット {args.dataset!r} を Weave から取得できません: {e}\n"
                 "notebooks/part3_1_eval.ipynb で登録してください")
    print(f"dataset: {dataset.name}:{dataset.ref.digest}  ({len(dataset.rows)} rows)")

    evaluation = weave.Evaluation(
        name=f"eval-{dataset.name}",
        dataset=dataset,
        scorers=[retrieval_hit, answer_correct],
        trials=args.trials,
        evaluation_name=args.label,
    )
    # どの抽出器・埋め込みモデルで作ったインデックスで測ったかを、Evaluation の呼び出しに記録する
    meta = index_meta(args.index)
    print(f"index: {meta}")
    with weave.attributes({"index": meta}):
        summary = asyncio.run(evaluation.evaluate(RagModel(
            index=args.index, top_k=args.top_k, use_collection=args.use_collection, search=args.search,
            fts_query=args.fts_query, fts_fields=args.fts_fields, rrf_k=args.rrf_k,
            rerank=args.rerank, rerank_depth=args.rerank_depth,
            reasoning_effort=args.reasoning_effort, select=args.select, score_threshold=args.score_threshold,
            min_selected=args.min_selected, max_selected=args.max_selected, expand=args.expand, max_back=args.max_back,
            max_context_chars=args.max_context_chars,
            **PROMPTS[args.prompt],
        )))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
