# day2 ハンズオン

## Advanced RAGインデックスの取得

講師から指定された日付のReleaseに添付された配布物を使います。教材ソースも同じ日付のものを用意してください。
`uv`（教材の要求する版）と `curl` が必要です。Python 3.13は `uv` が用意します。
取得にはGitHubへのログインやOpenAI APIキーは不要です。

`day2-starter/` の中で実行します。

```bash
./scripts/rag_indexes/download.sh --version 2026-10-08
```

`--version` を省略するとGitHubのLatest Releaseから取得します。研修では講師から指定された日付を使ってください。
指定したReleaseに配布物が添付されていない場合はエラーになります。

インデックスは `data/lancedb/` に配置されます。対象は `docs_basic`・`docs_structured`・`docs_vision` のテーブルと付随ファイルです。
`simple_rag` とその他のテーブルは残ります。対象テーブルを読んでいるStreamlit・Jupyterなどは停止してから取得し、配置後に再起動してください。

依存ライブラリの版が配布物と一致しない場合は、配置前にエラーになります。同じReleaseの教材ソースを使っているか確認してください。
