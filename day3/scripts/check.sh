#!/bin/bash
# completed/ を再生成し、コミット済みの内容から drift があれば失敗する。CI 用。
set -euo pipefail

cd "$(dirname "$0")/.."

./scripts/build.sh

if [ -n "$(git status --porcelain completed/)" ]; then
  echo "ERROR: completed/ is out of sync with starter + chapter overlays." >&2
  echo "Run scripts/build.sh and commit the result." >&2
  git status --short completed/ >&2
  exit 1
fi

echo "completed/ is up to date"
