"""Vision 抽出器(extractors/vision.py)と、配布するインデックスのマニフェスト(scripts/rag_indexes/validate.py)で共有する設定。

マニフェストにどのモデルで図の説明を作ったかを記録するので、OpenAI のクライアントを作らずに読める別ファイルにしてある。
"""

import os

VISION_MODEL = os.environ.get("VISION_MODEL", "gpt-6-luna")
VISION_REASONING_EFFORT = os.environ.get("VISION_REASONING_EFFORT", "low")
