const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

// Next.js の API route などアプリ内パスに basePath を付与する。
// <Link> や useRouter は Next.js が自動で basePath を付けるため、
// 用途は client-side の fetch / window.location など「素のパス」を扱う場面に限る。
export function apiPath(path: string): string {
  return `${BASE_PATH}${path}`;
}
