#!/usr/bin/env bash
# 講師用: 3種類のAdvanced RAGインデックスを生成・検証して配布用に梱包する。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec uv run --no-project --python 3.13 python "$script_dir/cli.py" create "$@"
