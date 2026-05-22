import re

import requests
import toml


def extract_pkg_name(spec: str) -> str:
    """
    バージョン指定（==,>=,<,!=,~= など）が入った文字列から
    パッケージ名部分だけを抜き出す
    """
    return re.split(r"[\s<>=!~]+", spec, 1)[0]


def fetch_latest_versions(packages):
    """
    PyPIから指定されたパッケージリストの最新バージョンを取得する
    """
    results = []
    for pkg in packages:
        resp = requests.get(f"https://pypi.org/pypi/{pkg}/json", timeout=5)
        if resp.ok:
            ver = resp.json()["info"]["version"]
            results.append(f'"{pkg}=={ver}",')
        else:
            results.append(f"{pkg}: failed to fetch")
    return results


def main() -> None:
    # pyproject.toml を読み込む
    py = toml.load("pyproject.toml")

    # 通常依存
    deps = py.get("project", {}).get("dependencies", [])
    # dev-dependencies（uv）
    dev_deps = py.get("tool", {}).get("uv", {}).get("dev-dependencies", [])

    # パッケージ名だけのリストに変換
    packages = [extract_pkg_name(s) for s in deps]
    dev_packages = [extract_pkg_name(s) for s in dev_deps]

    # 結果表示
    print("=== project.dependencies ===")
    for pkg in packages:
        print(pkg)
    print("\n=== tool.uv.dev-dependencies ===")
    for pkg in dev_packages:
        print(pkg)

    # PyPI から最新バージョンを取ってくる
    print("\n=== latest versions from PyPI (project.dependencies) ===")
    for result in fetch_latest_versions(packages):
        print(result)

    print("\n=== latest versions from PyPI (tool.uv.dev-dependencies) ===")
    for result in fetch_latest_versions(dev_packages):
        print(result)


if __name__ == "__main__":
    main()
