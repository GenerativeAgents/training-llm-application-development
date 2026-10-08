#!/usr/bin/env bash
# 受講者用: Releaseまたはローカルの配布物から3種類のインデックスを配置する。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec uv run --no-project --python 3.13 python "$script_dir/rag_indexes.py" download "$@"
