import Link from "next/link";

export function Header() {
  return (
    <header className="border-b bg-white">
      <div className="flex h-14 items-center justify-between px-6">
        <Link href="/" className="font-bold text-lg">
          株式会社サンプルエージェント
        </Link>
        <nav className="flex gap-6 text-sm">
          <Link href="/" className="text-gray-600 hover:text-gray-900">
            トップ
          </Link>
          <Link href="/admin" className="text-gray-600 hover:text-gray-900">
            管理画面
          </Link>
        </nav>
      </div>
    </header>
  );
}
