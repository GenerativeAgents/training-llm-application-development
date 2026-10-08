#!/usr/bin/env bash
# 講師用: 指定した日付の教材Release・添付ファイル・リモートタグを削除する。
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec uv run --no-project --python 3.13 python "$script_dir/cli.py" delete "$@"
