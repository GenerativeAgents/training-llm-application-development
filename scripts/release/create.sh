#!/usr/bin/env bash
# 講師用: 生成元コミットの日付タグと教材Releaseを作成し、インデックスを添付する。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec uv run --no-project --python 3.13 python "$script_dir/cli.py" create "$@"
