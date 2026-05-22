#!/usr/bin/env bash
#
# day2/ から day2-starter/ を再生成するスクリプト。
# day2/ を更新したらこれを実行して day2-starter/ を追従させる。
#
#   使い方: bash scripts/generate-day2-starter/main.sh
#           （make build からも実行される）
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SRC="$ROOT/day2"
DEST="$ROOT/day2-starter"
INCLUDE_FILE="$SCRIPT_DIR/include.txt"
OVERRIDES="$SCRIPT_DIR/overrides"

# 安全確認
[ -d "$SRC" ]          || { echo "ERROR: $SRC が見つかりません" >&2; exit 1; }
[ -f "$INCLUDE_FILE" ] || { echo "ERROR: $INCLUDE_FILE が見つかりません" >&2; exit 1; }
[ "$(basename "$DEST")" = "day2-starter" ] \
  || { echo "ERROR: 出力先が不正です: $DEST" >&2; exit 1; }

# 0. 出力先をまるごと作り直す（全面再生成）
rm -rf "$DEST"

# 1. include.txt に列挙したファイル/ディレクトリだけを day2-starter/ にコピー
#    （# 始まりの行と空行はコメントとして除去してから rsync に渡す）
#    __pycache__ 等は許可ディレクトリ内に紛れるため --exclude で併用除外する
#    --files-from 使用時は -a に -r が含まれないため -r を明示する
grep -vE '^[[:space:]]*(#|$)' "$INCLUDE_FILE" \
  | rsync -a -r --files-from=- \
      --exclude='__pycache__/' \
      --exclude='.ipynb_checkpoints/' \
      --exclude='.DS_Store' \
      "$SRC/" "$DEST/"

# 2. starter 専用の上書きファイルを適用
if [ -d "$OVERRIDES" ]; then
  rsync -a "$OVERRIDES/" "$DEST/"
fi

echo "OK: $DEST を $SRC から再生成しました"
