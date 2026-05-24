from typing import Any

import weave


@weave.op()
def classification_accuracy(output: dict[str, Any], expected_topic: str) -> dict[str, Any]:
    """トピック分類の正確性を評価する。

    ワークフローが出力した topic と、データセットの期待値を比較する。
    完全一致で 1、不一致で 0 を返す。
    """
    predicted = output["topic"]
    score = 1 if predicted == expected_topic else 0
    return {"key": "classification_accuracy", "score": score}
