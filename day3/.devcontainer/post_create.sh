#!/bin/bash
set -euo pipefail

# Install Git completion
echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc

# Keep support for web/.npmrc min-release-age when using Node.js 24.11.1
npm install --global npm@11.19.0

# Install Chromium for Playwright
npx --yes playwright install-deps chromium

# Install Claude Code (pinned version)
curl -fsSL https://claude.ai/install.sh | bash -s -- 2.1.220
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
