# Install Git completion
echo "source /usr/share/bash-completion/completions/git" >> ~/.bashrc

# Install Chromium for Playwright
npx --yes playwright install-deps chromium

# Install Claude Code
curl -fsSL https://claude.ai/install.sh | bash
