import type { NextConfig } from "next";

const basePath = process.env.NEXT_PUBLIC_BASE_PATH || undefined;
const allowedDevOrigins = process.env.ALLOWED_DEV_ORIGINS
  ?.split(",")
  .map((s) => s.trim())
  .filter(Boolean);

const nextConfig: NextConfig = {
  // Next.js 16.3 以降は next dev / next build が web/ 直下に AGENTS.md と CLAUDE.md を
  // 生成する。エージェント向けの指示は day3 直下の CLAUDE.md に集約しているので無効化する
  agentRules: false,
  ...(basePath ? { basePath } : {}),
  ...(allowedDevOrigins?.length ? { allowedDevOrigins } : {}),
};

export default nextConfig;
