# 環境構築

このリポジトリのソースコードを動かすための環境構築手順です

## 動作確認環境

Linux (Dev Container) で動作確認しています。

Windowsの場合、WSL 2でも動作する可能性が高いです。

> [!WARNING]
> WSL 2では、「/mnt/c」ディレクトリ以下ではうまく動作しない可能性があります。「/home/<ユーザ名>」ディレクトリ以下を使用するようにしてください。


## 前提条件

以下をインストールしてください。

- Git
- uv (https://docs.astral.sh/uv/getting-started/installation/)
- VSCode系のエディタ
- [../.vscode/extensions.json](../.vscode/extensions.json)に記載のVSCode拡張機能

## 環境構築手順

1. リポジトリのクローン
```console
git clone https://github.com/GenerativeAgents/training-llm-application-development.git
cd training-llm-application-development
```

2. VSCodeを起動
```console
code .
```

3. Pythonと依存パッケージのインストール
```console
uv sync
```

4. Streamlitアプリケーションの起動
```console
make streamlit
```
または
```console
uv run streamlit run app.py --server.port 8080
```
