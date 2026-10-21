FROM mcr.microsoft.com/devcontainers/base:bookworm

# Refresh the trust store before Dev Container features download any tools.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && dpkg --compare-versions "$(dpkg-query -W -f='${Version}' ca-certificates)" ge '20250419~deb12u2' \
    && rm -rf /var/lib/apt/lists/*
