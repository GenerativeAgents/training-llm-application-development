"""Amazon Bedrock AgentCore Gateway の Web Search Tool を呼び出すモジュール。

AWS が提供するマネージドな Web 検索（AgentCore Gateway の built-in connector）を
呼び出す。API キーは不要で、AWS の認証情報（ハンズオン環境では EC2 の IAM ロール）
で SigV4 署名して呼び出す。

- Gateway の URL は環境変数 ``AGENTCORE_GATEWAY_URL`` から読む
  （ハンズオン環境では code-server の環境変数として設定済み）。SigV4 の署名リージョンは
  この URL のホスト名から導出する。
- ツール名は環境変数 ``AGENTCORE_WEB_SEARCH_TOOL_NAME`` から読む（未設定なら既定値）。
- Gateway は MCP サーバーなので、``tools/call`` の JSON-RPC を HTTP POST する。
  MCP SDK は使わず ``requests`` で直接呼び出している。

このモジュールは Web 検索の関数 ``web_search`` を提供するだけで、LangChain には依存しない。
LangChain のツールや Retriever として使う場合は、呼び出し側でこの関数をラップする。

使い方::

    from app.tools.web_search import web_search

    results = web_search("東京の明日の天気", max_results=5)
"""

import json
import os
import random
import re
import time
from typing import Any, NotRequired, TypedDict
from urllib.parse import urlparse

import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

# クライアントが使う MCP プロトコルバージョン（ヘッダと _meta の組み立てがこの版に依存する）
_mcp_protocol_version = "2026-07-28"
# ツール名の既定値。"<Gateway Target 名>___<connector のツール名>" という形式になる
_default_tool_name = "web-search-tool___WebSearch"
# Web Search Tool のクォータ（既定 10 リクエスト/秒）を超えると HTTP 429 が返るため、
# 指数バックオフでリトライする
_max_retries = 4

_ENV_GATEWAY_URL = "AGENTCORE_GATEWAY_URL"
_ENV_TOOL_NAME = "AGENTCORE_WEB_SEARCH_TOOL_NAME"


class WebSearchResult(TypedDict):
    """Web 検索の 1 件分の結果。キー名は Gateway のレスポンスのまま。"""

    text: str  # 本文の抜粋
    url: str
    title: str
    publishedDate: NotRequired[str]  # ISO 8601 形式。結果によっては含まれない


def _gateway_url() -> str:
    url = os.environ.get(_ENV_GATEWAY_URL)
    if not url:
        raise RuntimeError(
            f"環境変数 {_ENV_GATEWAY_URL} が設定されていません。"
            "ハンズオン環境では自動で設定されます。ローカルで実行する場合は、"
            "AgentCore Gateway の MCP エンドポイント URL を設定してください。"
        )
    return url


def _tool_name() -> str:
    return os.environ.get(_ENV_TOOL_NAME) or _default_tool_name


def _region_from_url(url: str) -> str:
    """Gateway URL のホスト名（<id>.gateway.bedrock-agentcore.<region>.amazonaws.com）から
    SigV4 の署名リージョンを取り出す。"""
    host = urlparse(url).hostname or ""
    match = re.search(r"\.bedrock-agentcore\.([a-z0-9-]+)\.", host)
    if match is None:
        raise RuntimeError(
            f"Gateway URL からリージョンを判定できません: {url}"
            "（<id>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp の形式を想定）"
        )
    return match.group(1)


def _signed_headers(url: str, body: bytes, headers: dict[str, str]) -> dict[str, str]:
    """AWS の認証情報でリクエストに SigV4 署名を付けたヘッダーを返す。"""
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        raise RuntimeError(
            "AWS の認証情報が見つかりません。"
            "ハンズオン環境では EC2 の IAM ロールが自動で使われます。"
        )
    aws_request = AWSRequest(method="POST", url=url, data=body, headers=headers)
    SigV4Auth(
        credentials.get_frozen_credentials(), "bedrock-agentcore", _region_from_url(url)
    ).add_auth(aws_request)
    return dict(aws_request.headers)


def _parse_response(response: requests.Response) -> dict[str, Any]:
    """JSON-RPC のレスポンスを dict にする。SSE（text/event-stream）で返る場合にも対応する。"""
    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("text/event-stream"):
        data_lines = [
            line[len("data:") :].strip()
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            raise RuntimeError(
                f"Gateway から空の SSE レスポンスが返りました: {response.text}"
            )
        return json.loads(data_lines[-1])
    return response.json()


def web_search(query: str, max_results: int = 5) -> list[WebSearchResult]:
    """Web 検索を実行し、検索結果のリストを返す。

    Args:
        query: 検索クエリ（200 文字以内。超えた分は切り捨てる）
        max_results: 取得する件数（1〜25）
    """
    query = query[:200]
    max_results = max(1, min(25, max_results))

    url = _gateway_url()
    tool_name = _tool_name()
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": {"query": query, "maxResults": max_results},
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": _mcp_protocol_version,
                "io.modelcontextprotocol/clientInfo": {
                    "name": "training-llm-application-development",
                    "version": "1.0.0",
                },
                "io.modelcontextprotocol/clientCapabilities": {},
            },
        },
    }
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": _mcp_protocol_version,
        "Mcp-Method": "tools/call",
        "Mcp-Name": tool_name,
    }

    for attempt in range(_max_retries + 1):
        response = requests.post(
            url,
            data=body,
            headers=_signed_headers(url, body, headers),
            timeout=60,
        )
        if response.status_code == 429 and attempt < _max_retries:
            # 全受講者で共有するクォータを超えたので、少し待って再試行する
            time.sleep(0.5 * (2**attempt) + random.uniform(0, 0.5))
            continue
        break

    if response.status_code != 200:
        raise RuntimeError(
            f"Web 検索に失敗しました (HTTP {response.status_code}): {response.text}"
        )

    data = _parse_response(response)
    if "error" in data:
        raise RuntimeError(f"Web 検索に失敗しました: {data['error']}")

    result = data["result"]
    if result.get("isError"):
        raise RuntimeError(f"Web 検索に失敗しました: {result.get('content')}")

    # 結果は structuredContent（対応バージョンの場合）か、content[0].text の JSON 文字列に入る
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and "results" in structured:
        return structured["results"]
    text = result["content"][0]["text"]
    return json.loads(text)["results"]
