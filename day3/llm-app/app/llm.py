import os
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

load_dotenv()

# 既定では Amazon Bedrock の Claude を使う。ハンズオン環境では EC2 の IAM ロールで認証するので API キーは不要。
# Bedrock に障害があるときのバックアップとして、.env に ANTHROPIC_API_KEY を書くと Anthropic API を直接使う。
_USE_ANTHROPIC_API: bool = bool(os.environ.get("ANTHROPIC_API_KEY"))

BEDROCK_MODEL = "jp.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_REGION: str = os.environ.get("AWS_REGION") or "ap-northeast-1"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

_THINKING_CONFIG: dict[str, Any] = {"type": "enabled", "budget_tokens": 10000}
_THINKING_MAX_TOKENS = 16000


def get_model(*, thinking: bool = False) -> BaseChatModel:
    """Create a chat model instance, optionally with extended thinking enabled."""
    kwargs: dict[str, Any]
    if _USE_ANTHROPIC_API:
        kwargs = {
            "model": ANTHROPIC_MODEL,
            "model_provider": "anthropic",
        }
    else:
        kwargs = {
            "model": BEDROCK_MODEL,
            "model_provider": "anthropic_bedrock",
            "region_name": BEDROCK_REGION,
        }
    if thinking:
        kwargs["thinking"] = _THINKING_CONFIG
        kwargs["max_tokens"] = _THINKING_MAX_TOKENS
    return init_chat_model(**kwargs)  # type: ignore[no-any-return]
