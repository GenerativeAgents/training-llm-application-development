"""Vision抽出器と配布物の記録で共有する設定。APIクライアントは作成しない。"""

import os

VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-6-luna")
VISION_REASONING_EFFORT = os.environ.get("VISION_REASONING_EFFORT", "low")
