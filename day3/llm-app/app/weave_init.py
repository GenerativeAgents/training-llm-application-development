import os

import weave
from weave.trace.weave_client import WeaveClient

_TEMPLATE_VALUE = "training-ai-agent-dev-"


def init_weave() -> WeaveClient:
    """WANDB_PROJECT 環境変数を読み込んで weave.init を呼び出す."""
    project = os.environ["WANDB_PROJECT"]
    assert project != _TEMPLATE_VALUE, (
        "WANDB_PROJECT を training-ai-agent-dev-<自分の名前> の形式で設定してください"
    )
    return weave.init(project)
