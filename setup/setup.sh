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

# Claude Codeがインストールされるパスを追加
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# Claude CodeをANTHROPIC_API_KEYで認証するための設定を追加
# 参考: https://www.reddit.com/r/ClaudeAI/comments/1jwvssa/comment/mtt0urz/
mkdir -p ~/.claude
cat << 'EOF' > ~/.claude/settings.json
{
  "apiKeyHelper": "grep '^ANTHROPIC_API_KEY=' .env | cut -d'=' -f2-"
}
EOF

# Claude CodeのCPU・メモリの制限を設定
cat << 'EOF' >> ~/.bashrc
claude() {
  sudo systemd-run --quiet --pty \
    -p "User=$(id -un)" -p "Group=$(id -gn)" \
    -p "WorkingDirectory=$PWD" \
    -p "CPUQuota=${CLAUDE_CPU_QUOTA:-100%}" \
    -p "MemoryMax=${CLAUDE_MEM_MAX:-2G}" \
    -- /bin/bash -ic 'exec "$(command -v claude)" "$@"' -- "$@"
}
EOF

# code-serverの設定を追加
mkdir -p ~/.local/share/code-server/User
cat << 'EOF' > ~/.local/share/code-server/User/settings.json
{
    "editor.stickyScroll.enabled": false,
    "terminal.integrated.stickyScroll.enabled": false,
    "terminal.integrated.gpuAcceleration": "off"
}
EOF

# Claude Codeのインストール
curl -fsSL https://claude.ai/install.sh | bash

# Visual Studio Codeの拡張機能のインストール
recommendations=(
  "charliermarsh.ruff"
  "ms-python.mypy-type-checker"
  "ms-toolsai.jupyter"
)
for recommendation in "${recommendations[@]}"; do
  code-server --install-extension "${recommendation}" --force
done
