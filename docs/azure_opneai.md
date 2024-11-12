# Azure OpenAI Service の利用手順

万が一 OpenAI API の障害の場合に、OpenAI API の代替として Azure OpenAI Service を使用する手順です。

## .env ファイル

.env ファイル内の OPENAI_API_KEY を削除またはコメントアウトします。

```
# OPENAI_API_KEY=
```

.env ファイルに Azure OpenAI Service の環境変数を記述します。

```
AZURE_OPENAI_ENDPOINT=https://example-endpoint.openai.azure.com
AZURE_OPENAI_API_KEY=
```

## openai パッケージ

openai パッケージを使用する箇所は、以下のように変更します。

変更前

```python
from openai import OpenAI
client = OpenAI()
```

変更後

```python
import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version="2024-08-01-preview",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)
```

> [!WARNING]
> client を実行する際の model パラメータには、Azure OpenAI Service で設定したデプロイ名を設定します。

## LangChian の ChatOpenAI

LangChain の ChatOpenAI を使用する箇所は、以下のように変更します。

変更前

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

変更後

```python
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2024-08-01-preview",
    temperature=0,
)
```

> [!WARNING]
> azure_deployment には、Azure OpenAI Service で設定したデプロイ名を設定します。

## LangChian の OpenAIEmbeddings

LangChain の OpenAIEmbeddings を使用する箇所は、以下のように変更します。

変更前

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

変更後

```python
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small")
```

> [!WARNING]
> model には、Azure OpenAI Service で設定したデプロイ名を設定します。
