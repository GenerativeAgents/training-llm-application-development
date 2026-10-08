# day2 ハンズオン

## 受講者: Advanced RAGインデックスの取得

教材ソースと同じ日付のReleaseに添付された配布物を使います。`uv`（教材の要求する版）と `curl` が必要です。
GitHubへのログインやOpenAI APIキーは取得には不要です。Python 3.13は `uv` が用意します。

`day2-starter/` の中で実行します。

```bash
./scripts/rag_indexes/download.sh --version 2026-10-08
uv sync --frozen
```

`--version` を省略すると、GitHubのLatest Releaseから取得します。Draft・Prereleaseは対象外です。
Latestに配布物がまだ添付されていない場合はエラーになります。研修では日付を指定すると教材の版を固定できます。

リポジトリ全体を開いている場合は、ルートのスクリプトを使えます。

```bash
# 完全版のday2/data/lancedb/へ配置
./scripts/rag_indexes/download.sh --version 2026-10-08

# starterへ配置
./scripts/rag_indexes/download.sh --version 2026-10-08 --destination day2-starter/data/lancedb
```

取得・展開・生成条件の確認に成功してから、次の9項目だけを交換します。

```text
data/lancedb/
├── docs_basic.lance/
├── docs_basic.jsonl
├── docs_basic.meta.json
├── docs_structured.lance/
├── docs_structured.jsonl
├── docs_structured.meta.json
├── docs_vision.lance/
├── docs_vision.jsonl
├── docs_vision.meta.json
├── rag_indexes_manifest.json       # 生成元・生成条件
└── rag_indexes_attributions/       # 原資料の出典・ライセンス
```

`simple_rag.lance/` とその他のテーブルは残ります。配布物のLanceDB・PyArrow・Sudachi・辞書の版が配置先の
`uv.lock` と一致しない場合は、交換前にエラーになります。生成時のuvの版も表示します。
対象テーブルを読んでいるStreamlit・Jupyterなどは停止してから実行し、配置後に再起動してください。

## 講師: 配布物の生成

以下はリポジトリのルートで実行します。Git、uv、公開操作にはGitHub CLI（`gh auth login` 済み）が必要です。
`day2/.env` に `OPENAI_API_KEY` と `WANDB_PROJECT` を設定してください（`.env.template` を参照）。
インデックス生成ではOpenAI APIを利用します。Visionのモデルと推論設定は `VISION_MODEL`・
`VISION_REASONING_EFFORT` で指定でき、既定値は抽出器の設定に従います。

1. ソース変更をコミットし、starterも最新にしてGitHubへpushします。未コミット・未追跡のファイルがある場合は生成を開始しません。
2. 3種類のインデックスを生成します。

```bash
bash scripts/generate-day2-starter/main.sh
# starterの差分を確認してコミット・pushしてから生成
./scripts/rag_indexes/create.sh
```

各抽出器は `day2/data/lancedb/` の対応テーブルを再生成します。`simple_rag` は触りません。
埋め込みモデルは既定で `text-embedding-3-small` です。変更する場合は
`--embedding-model text-embedding-3-large` を指定し、検索側も同じモデルにします。
講師の `day2/data/cache/vision/` は次回の生成に利用できます。

スクリプトはロックされた依存関係を `uv sync --frozen` で用意し、3テーブルの件数・スキーマ・ベクトル検索・
全文検索・メタデータ絞り込みを確認します。梱包後にも一時領域へ復元して同じ検索検証を行います。
配布物は `dist/advanced-rag-indexes.tar.gz` に出力します。`--archive PATH` で出力先を変更できます。

配布物には対象9項目のディレクトリ／ファイル全体と、生成条件の `manifest.json`、原資料の出典・ライセンスREADMEを含めます。
元のコーパス、Visionキャッシュ、`simple_rag` は含めません。チェックサムファイルは生成しません。
生成元コミット、Python・uv・主要依存の版、埋め込みモデル、Vision設定、件数・ベクトル次元を記録します。
生成中はソース・コーパス・対象テーブルを別プロセスから変更しないでください。

## 講師: ローカルとDraftでのテスト

starterや別のクリーンな受講環境に配置し、ハンズオンのアプリ動作も確認します。

```bash
./scripts/rag_indexes/download.sh --archive dist/advanced-rag-indexes.tar.gz \
  --destination day2-starter/data/lancedb

# Draftを作成して添付。まだ一般公開されない
./scripts/release/create.sh --version 2026-10-08 --draft

# Draftはアクセス権のある講師がghで取得し、同じ復元処理を試す
mkdir -p tmp/release-test
gh release download 2026-10-08 \
  --repo GenerativeAgents/training-llm-application-development \
  --pattern advanced-rag-indexes.tar.gz --dir tmp/release-test --clobber
./scripts/rag_indexes/download.sh --archive tmp/release-test/advanced-rag-indexes.tar.gz \
  --destination day2-starter/data/lancedb
```

## 講師: 正式Releaseの作成・公開

検証した配布物を使い、1コマンドで日付タグ・教材Release・添付を作成します。

```bash
./scripts/release/create.sh --version 2026-10-08
```

タグは配布物に記録した生成元コミットを指します。GitHubにpush済みのコミットが必要です。
`--version` は必須です。同名タグが別コミットを指す場合や、同名Releaseが公開済みの場合はエラーになります。
既存Draftが同じ生成元なら添付を更新し、そのDraftを公開します。初回もDraftへ添付してから公開するので、
添付に失敗した場合はDraftのまま残り、同じコマンドで再試行できます。
公開時はLatestに指定します。教材ソースは同じタグのGitHub標準のソースアーカイブから取得できます。

公開後に認証なしで `scripts/rag_indexes/download.sh --version ...` を実行して取得を確認してください。
`--repo owner/repo` はテスト用の別リポジトリに向ける場合に使えます。
配布物が別のパスにある場合は `--archive PATH` で指定します（Release上の添付名は固定です）。

## 講師: 検証用Releaseの削除

```bash
./scripts/release/delete.sh --version 2026-10-08
```

指定日付のRelease・全添付・リモートのGitタグを削除します。ソースコミットとローカルのタグは残ります。
`--version` は必須で、Latestを自動で選ぶ動作はありません。通常は `gh` の確認が入り、
`--yes` で確認を省略できます。公開配布済みの教材は保持し、検証用の削除に使います。
同じ日付で作り直す場合は、このスクリプトでリモートタグも削除してから作成してください。

## スクリプトの保守

配布用の処理をリポジトリの `scripts/` 以下にまとめています。

```text
scripts/
├── rag_indexes/
│   ├── create.sh
│   ├── download.sh
│   ├── cli.py
│   ├── validate.py
│   └── tests/
└── release/
    ├── create.sh
    └── delete.sh
```

4つの `.sh` は共通処理の `scripts/rag_indexes/cli.py` を呼びます。取得・復元・Release操作にはPython標準ライブラリだけを使い、
生成時の検索検証は `scripts/rag_indexes/validate.py` がday2のuv環境で行います。
starterは生成スクリプトが `scripts/rag_indexes/download.sh` と `cli.py` をコピーします。starter側を直接編集しないでください。
starter生成時にはローカルの `data/lancedb/`・`data/cache/` をコピーしません。

外部API・実際のRelease操作なしで、配布・復元とRelease操作のテストを実行できます（リポジトリのルートから）。

```bash
uv run --project day2 --frozen python -m unittest discover -s scripts/rag_indexes/tests -v
```
