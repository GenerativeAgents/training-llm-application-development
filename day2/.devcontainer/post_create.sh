#!/bin/bash
set -euo pipefail

# Install Git completion
echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc

# Install Chromium dependencies for Playwright (same as training-shared ai-agent-dev)
npx --yes playwright install-deps chromium

# Install Claude Code (same version as training-shared common-setup.sh)
curl -fsSL https://claude.ai/install.sh | bash -s -- 2.1.220
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

uv sync
