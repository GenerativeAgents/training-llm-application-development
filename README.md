# AIエージェント開発者養成講座

AIエージェント開発者養成講座で使用するソースコードです。

ハンズオン環境の構築手順は[こちら](./setup/README.md)を参照してください。

## リポジトリ構成

講座の日程ごとにディレクトリを分けています。各日程には、講座中に参照するための完全版と、受講者に配布する自動生成版（starter）があります。

| ディレクトリ    | 役割                                                                                 |
| --------------- | ------------------------------------------------------------------------------------ |
| `day2/`         | day2 の完全版                                                                        |
| `day2-starter/` | day2 の受講者配布用 starter（**自動生成**）。直接編集しないこと。                    |
| `day3/`         | day3 の完全版                                                                        |
| `day3-starter/` | day3 の受講者配布用 starter（**自動生成**）。直接編集しないこと。                    |
| `setup/`        | ハンズオン環境の構築（AWS EC2 + code-server）。詳細は `setup/README.md` を参照。     |
| `docs/`         | 講座準備用ドキュメントと API キー取得ガイド（Cohere、Tavily、Weave、Azure OpenAI）。 |
| `scripts/`      | starter 生成用スクリプト（下記参照）。                                               |

## starter の生成

`day2-starter/` と `day3-starter/` は、`day2/` と `day3/` から**生成**されます。starter を手作業で編集しないでください。次回の `make build` でディレクトリごと上書き（`rm -rf`）されます。

```bash
make build   # day2/ と day3/ から両方の starter を再生成する
```

受講者に配布する内容を変更するには、以下の方法があります。

- **元ファイルを編集する**（`day2/` または `day3/`）。編集後に `make build` を実行します。
- **含めるファイルを制御する**：`scripts/generate-day{2,3}-starter/include.txt` で指定します。ここに記載されたパスのみが starter にコピーされ、それ以外は除外されます。演習用ファイルを受講者から隠す（例：day3 では `evals/__init__.py` のみを含め、残りは受講者に実装させる）のもこの仕組みです。
- **starter 用にファイルを上書きする**：`scripts/generate-day{2,3}-starter/overrides/` に対象パスと同じ構造で配置します。上書きは（include コピーの後に）最後に適用され、通常は解答コードを空にして受講者が埋められるようにします（例：`day3-starter` の `generate_response.py`）。

元ファイルを編集したら `make build` を実行し、starter の差分を確認してからコミットしてください。
