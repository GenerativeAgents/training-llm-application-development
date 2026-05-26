#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

if [ "$(whoami)" != "ubuntu" ]; then
  echo "This script must be run as ubuntu user" >&2
  exit 1
fi

# ソースコードのダウンロード
cd /home/ubuntu/environment
if [ ! -d "training-llm-application-development" ]; then
  git clone https://github.com/GenerativeAgents/training-llm-application-development.git
fi
cd training-llm-application-development

# uvのインストール
curl -LsSf https://astral.sh/uv/0.4.14/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
uv --version

# PythonとPythonパッケージのインストール
cd day2-starter
uv sync
uv run python --version

# Node.jsのインストール
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
nvm install 24.11.1

# Visual Studio Codeの拡張機能のインストール
recommendations=(
  "charliermarsh.ruff"
  "ms-python.mypy-type-checker"
  "ms-toolsai.jupyter"
)
for recommendation in "${recommendations[@]}"; do
  code-server --install-extension "${recommendation}" --force
done
