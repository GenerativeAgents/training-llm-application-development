#!/usr/bin/env bash
#
# day3/ から day3-starter/ を再生成するスクリプト。
# day3/ を更新したらこれを実行して day3-starter/ を追従させる。
#
#   使い方: bash scripts/generate-day3-starter/main.sh
#           （make build からも実行される）
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SRC="$ROOT/day3"
DEST="$ROOT/day3-starter"
INCLUDE_FILE="$SCRIPT_DIR/include.txt"
OVERRIDES="$SCRIPT_DIR/overrides"

# 安全確認
[ -d "$SRC" ]          || { echo "ERROR: $SRC が見つかりません" >&2; exit 1; }
[ -f "$INCLUDE_FILE" ] || { echo "ERROR: $INCLUDE_FILE が見つかりません" >&2; exit 1; }
[ "$(basename "$DEST")" = "day3-starter" ] \
  || { echo "ERROR: 出力先が不正です: $DEST" >&2; exit 1; }

# 0. 出力先をまるごと作り直す（全面再生成）
rm -rf "$DEST"

# 1. include.txt に列挙したファイル/ディレクトリだけを day3-starter/ にコピー
#    （# 始まりの行と空行はコメントとして除去してから rsync に渡す）
#    ディレクトリ単位コピーで紛れ込み得る生成物・ローカル設定は --exclude で除外する
#    --files-from 使用時は -a に -r が含まれないため -r を明示する
grep -vE '^[[:space:]]*(#|$)' "$INCLUDE_FILE" \
  | rsync -a -r --files-from=- \
      --exclude='__pycache__/' \
      --exclude='.mypy_cache/' \
      --exclude='.ruff_cache/' \
      --exclude='.venv/' \
      --exclude='node_modules/' \
      --exclude='.next/' \
      --exclude='.ipynb_checkpoints/' \
      --exclude='.DS_Store' \
      --exclude='settings.local.json' \
      --exclude='tsconfig.tsbuildinfo' \
      --exclude='next-env.d.ts' \
      "$SRC/" "$DEST/"

# 2. starter 専用の上書きファイルを適用
if [ -d "$OVERRIDES" ]; then
  rsync -a "$OVERRIDES/" "$DEST/"
fi

echo "OK: $DEST を $SRC から再生成しました"
