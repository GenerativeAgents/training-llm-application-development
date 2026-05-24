#!/bin/bash
# starter/ を rsync で completed/ に同期し、chapter07..09 を overlay する。
#
# completed/ で Dev Container や web (Next.js) / llm-app (uvicorn) を起動した
# まま本スクリプトを実行しても壊れないように、以下の方針をとる:
#   - completed/ ディレクトリ自体は削除せず inode を維持する
#     （Dev Container のバインドマウントが生き続ける）。
#   - starter/ の同期には rsync を使い、dev server / ランタイムが作る
#     成果物（node_modules / .venv / .next / SQLite DB / 各種キャッシュ / .env 等）
#     を --exclude で明示的に保護する。--exclude は転送対象からも --delete
#     の削除対象からも外れるため、completed/ で起動中のサービスの
#     ランタイム状態を壊さずに starter の差分だけを反映できる。
#   - chapter07/08/09 は各レイヤの git 管理ファイルを overlay コピー。
#
# macOS 標準の openrsync は dir-merge フィルタ（`:- .gitignore`）を
# サポートしないため、保護対象は明示列挙している。.gitignore に新しい
# 成果物パターンを足したときは、必要に応じて下の --exclude も同期させる。
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p completed

# 1) starter → completed を rsync で同期。
rsync -a --delete \
  --exclude='/web/node_modules' \
  --exclude='/web/.next' \
  --exclude='/web/data/*.db' \
  --exclude='/web/data/*.db-shm' \
  --exclude='/web/data/*.db-wal' \
  --exclude='/llm-app/.venv' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.mypy_cache' \
  --exclude='.ruff_cache' \
  --exclude='.env' \
  starter/ completed/

# 2) chapter07/08/09 を overlay。
OVERLAYS=(chapter07 chapter08 chapter09)
for layer in "${OVERLAYS[@]}"; do
  if [ ! -d "$layer" ]; then
    echo "layer not found: $layer" >&2
    exit 1
  fi
  (
    cd "$layer"
    git ls-files -co --exclude-standard
  ) | while IFS= read -r rel; do
    dest="completed/$rel"
    mkdir -p "$(dirname "$dest")"
    cp -p "$layer/$rel" "$dest"
  done
done

echo "synced completed/ from starter + ${OVERLAYS[*]}"
