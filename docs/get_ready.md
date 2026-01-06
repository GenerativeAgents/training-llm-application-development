# 講座で使用するサービスの事前準備のお願い

講座内で次のサービスを使用します。これらのサービスが利用できるよう、事前に次の準備をお願いいたします。

|     | サービス名        | 講座での利用内容                                                                                                                                                                                                                                 | 実施いただく事前準備<br />（後述）             |
| --- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| ①   | LangSmith         | 開発したアプリケーションのトレースや評価に使用します。                                                                                                                                                                                           | 社内申請<br />アカウント登録<br />API-KEY 発行 |
| ②   | Tavily            | アプリケーションを Web 検索と連携させる際に使用します。                                                                                                                                                                                          | 社内申請<br />アカウント登録<br />API-KEY 発行 |
| ③   | Cohere            | RAG アプリケーションのリランク処理に使います。                                                                                                                                                                                                   | 社内申請<br />アカウント登録<br />API-KEY 発行 |
| ④   | GitHub            | 見本となるソースコードの共有で使います。                                                                                                                                                                                                         | 社内申請<br />接続確認                         |
| ⑤   | 弊社講座用サイト  | 小テストやアンケートで使用します。                                                                                                                                                                                                               | 社内申請<br />接続確認                         |
| ⑥   | ハンズオン環境    | ブラウザで利用できる VSCode の環境です。                                                                                                                                                                                                         | 社内申請<br />接続確認                         |
| ⑦   | Zoom ミーティング | オンライン開催の場合はミーティングのツールとして使います。オフライン開催の場合でも、講師と受講者間でのファイルや URL などのやり取りのために、チャット機能のみを使います。<br />※個社開催の場合は別のツールへの変更も可能ですのでご相談ください。 | 社内申請                                       |

## 1. 各サービスを利用するための社内申請（必要な場合）

① から ⑦ のサービスの利用にあたり、社内で利用申請やプロキシ設定などが必要な場合は手続きをお願いします。

### ネットワーク接続に関する補足

- ① から ③ のサービスは、後述のアカウント登録と API-KEY 発行が実施できれば問題ありません。API 利用時は弊社が AWS にて用意するハンズオン環境からの直接アクセスになるため、貴社のネットワークは経由いたしません。
- ④ は、このページが閲覧できていれば問題ありません。
- ⑤、⑥ は、後述の接続確認が実施できれば問題ありません。
- ⑦ の Zoom ミーティングは個社開催の場合に限り別のツールへの変更も可能です。変更が必要な場合はご相談ください。

## 2. アカウント登録と API-KEY 発行

① から ③ のサービスについては、事前にアカウントの登録と API-KEY の発行をお願いします。

### ①LangSmith

以下の手順で、LangSmith のアカウント登録と API-KEY 発行ができます。

1. [https://www.langchain.com/langsmith/observability](https://www.langchain.com/langsmith/observability)にアクセスし、右上の「Try LangSmith」をクリックします。

![alt text](images/smith_try.png)

2. Google アカウントなどで連携するか、メールアドレスと任意のパスワードを入力して「Continue」を選びます。メールアドレスの場合、パスワードは画面に表示される複雑さを満たす必要があります。

![alt text](images/smith_create_account.png)

3. メールアドレスの場合は確認メールが届くので、メール内の「Confirm Email Address」をクリックします。LangSmith のサイトに戻るので「Confirm」をクリックします。

![alt text](images/smith_confirm.png)

4. いくつかアンケートがあるので回答します。画面下部の「Skip」でスキップすることもできます。

![alt text](images/smith_survey.png)

5. アンケートが終わるとトップ画面になりますので、左下の「Settings」をクリックします。

![alt text](images/smith_settings.png)

6. API-KEY の設定画面になるので、右上の「+ API Key」をクリックします。

![alt text](images/smith_settings_top.png)

7. Description にはこの API-KEY の任意の説明を入力し、Key Type で「Personal Access Token」、Default Workspace で「Workspace 1」を選択して、「Create API Key」をクリックします。

![alt text](images/smith_create_api_key.png)

8. API-KEY が発行されるので「Copy」をクリックしてコピーし、講義まで保存しておいてください。「I've saved the API key to a safe place」をクリックすれば前の画面に戻り、API-KEY の発行作業は終了です。

![alt text](images/smith_api_key.png)

### ②Tavily

以下の手順で、Tavily のアカウント登録と API-KEY の発行ができます。

1. [https://www.tavily.com/](https://www.tavily.com/)にアクセスし、右上の「Sign Up」をクリックします。

![alt text](images/tavily_top.png)

2. ログイン画面が出てくるので、入力せずに一番下の「Don't have an account?」の横にある「Sign up」をクリックします。

![alt text](images/tavily_welcome.png)

3. アカウント作成画面になるので、Google アカウントなどで連携するか、メールアドレスと画像の文字を入力して「Continue」を選びます。

![alt text](images/tavily_create_account.png)

4. メールアドレスの場合は設定するパスワードを入力します。画面に表示される複雑さを満たす必要があります。その後、確認メールが届くので、メール内の「Verify Your Account」をクリックします。

![alt text](images/tavily_password.png)

5. Tavily のサイトで最初のメッセージが表示されるので、確認して閉じます。

![alt text](images/tavily_message.png)

6. 自動的に API-KEY が発行されるので、コピーのアイコンをクリックしてコピーし、講義まで保存しておいてください。

![alt text](images/tavily_api_key.png)

### ③Cohere

以下の手順で、Cohere のアカウント登録と API-KEY の発行ができます。

1. [https://cohere.com/](https://cohere.com/)にアクセスし、右上の「Sign in」をクリックします。

![alt text](images/cohere_top.png)

2. ログイン画面が出てくるので、入力せずに一番下の「New user?」の横にある「Sign up」をクリックします。

![alt text](images/cohere_login.png)

3. アカウント作成画面になるので、Google アカウントなどで連携するか、メールアドレスと設定するパスワードを入力して「Sign up」を選びます。メールアドレスの場合、パスワードは画面に表示される複雑さを満たす必要があります。また、確認メールが届くので、メール内の「Confirm your email」をクリックします。

![alt text](images/cohere_create_account.png)

4. 名前を尋ねられるので、入力して「Continue」をクリックします。

![alt text](images/cohere_name.png)

5. いくつかアンケートがあるので回答します。右下の「Skip this step」でスキップすることもできます。

![alt text](images/cohere_survey.png)

6. ダッシュボードが表示されるので、左のメニューから「API Keys」をクリックします。

![alt text](images/cohere_dashboard.png)

7. トライアル用の API-KEY が発行されているので、目のアイコンをクリックします。

![alt text](images/cohere_api_keys.png)

8. コピーのアイコンをクリックしてコピーし、講義まで保存しておいてください。

![alt text](images/cohere_api_key.png)

## 3. 接続確認

### ⑤ 弊社講座用サイト

次の URL にアクセスし、「サイトに正常に接続できました」と表示されれば正常です。

https://academy.generative-agents.co.jp/

### ⑥ ハンズオン環境

ハンズオン環境は、以下の URL でご提供します。接続確認用の URL は、別途メールにてご案内します。

https://<受講者ごとに異なるランダムな文字列>.cloudfront.net

<hr>

ご不明な点がありましたら、ご案内メールの返信にてお知らせください。
