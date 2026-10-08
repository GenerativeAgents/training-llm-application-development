# 厚生労働省の施策資料 — 出典とライセンス

このフォルダには、厚生労働省が公開している次の資料を収録しています。

| ファイル | 原本 | 掲載ページ |
| --- | --- | --- |
| `生涯現役促進地域連携事業の概要.pptx` | https://www.mhlw.go.jp/content/000505085.pptx | [生涯現役促進地域連携事業について](https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/koyou_roudou/koyou/koureisha/koureisha-koyou_00005.html) |
| `健康日本21（第二次）参考資料スライド集.pptx` | https://www.mhlw.go.jp/bunya/kenkou/dl/kenkounippon21_sura.pptx | [健康日本21（第二次）](https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/kenkou_iryou/kenkou/kenkounippon21.html)の「普及啓発用資料」 |
| `令和8年度概算要求の概要（老健局）の参考資料.pdf` | https://www.mhlw.go.jp/wp/yosan/yosan/26syokan/dl/gaiyo-12-2.pdf | [令和８年度各部局の概算要求](https://www.mhlw.go.jp/wp/yosan/yosan/26syokan/03.html)の「老健局 [参考資料]」 |

- 作成者: 厚生労働省
- ライセンス: [公共データ利用規約(第1.0版)(PDL1.0)](https://www.digital.go.jp/resources/open_data/public_data_license_v1.0)。CC BY 4.0 と互換です
  - [厚生労働省ホームページ利用規約](https://www.mhlw.go.jp/chosakuken/index.html)による
- 改変: 内容は原本のままです。ファイル名だけ、内容が分かるものに変更しています
- 「健康日本21（第二次）参考資料スライド集」は平成25年3月末時点の資料で、掲載ページにも「各種データは最新のものとは限りません」と注記されています

## これらの資料から作ったデータ

本リポジトリの研修教材では、これらの資料から次のデータを作成しています。

- 検索用インデックス(`data/lancedb/`): 原本から抽出したテキストと、図形・グラフ・画像を AI で説明させた文
- 評価用データセット(`data/datasets/`)の質問・正解のうち、これらの資料の内容に基づくもの

## 利用・再配布するときのお願い

- 資料ごとに出典を記載してください。原本を加工したデータを使うときは、加工したことも示してください。このファイルを同梱すれば足ります。
  記載例: 「出典: 厚生労働省「健康日本21（第二次）参考資料スライド集」(https://www.mhlw.go.jp/bunya/kenkou/dl/kenkounippon21_sura.pptx)を加工して作成」
- 加工したデータを、厚生労働省が作成したかのような形で公表しないでください
