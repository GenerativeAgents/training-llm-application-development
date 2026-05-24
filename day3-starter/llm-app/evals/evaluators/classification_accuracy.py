from typing import Any


def classification_accuracy(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
    """トピック分類の正確性を評価する。

    ワークフローが出力した topic と、データセットの期待値を比較する。
    完全一致で 1.0、不一致で 0.0 を返す。
    """
    predicted = outputs["topic"]
    expected = reference_outputs["topic"]
    score = 1 if predicted == expected else 0
    return {"key": "classification_accuracy", "score": score}
