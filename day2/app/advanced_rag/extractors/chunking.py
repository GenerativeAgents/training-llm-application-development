"""位置単位(シート / スライド / ページ)の行を、埋め込み用のチャンクに切る共通部品。

どう切るかは抽出器の責任で、各抽出器の extract() が最後にここを呼ぶ(自前の切り方に替えてもよい)。
チャンクの ID(`#位置単位#連番`)は ingest.py が付けるので、ここでは本文だけを返す。
"""

from . import Chunked, Line, Section

MAX_CHARS = 3000  # 1 文書の目安。text-embedding-3-small の上限(8191 トークン)に十分収まる


def split_chunks(lines: list[Line]) -> list[str]:
    """MAX_CHARS を目安に分割する。半分を超えていれば見出し行で切る。

    2 つ目以降のチャンクには、直前の見出し行と表のヘッダ行を先頭に付け直す。
    """
    return [text for text, _ in split_chunks_with_index(lines)]


def split_chunks_with_index(lines: list[Line]) -> list[tuple[str, list[int]]]:
    """split_chunks と同じ分割で、各チャンクに入った元の行の番号も返す(付け直した見出し・ヘッダは含めない)。

    行ごとの付帯情報(PDF のページ番号など)から、チャンクごとの範囲を出すのに使う。
    """
    chunks: list[tuple[str, list[int]]] = []
    current: list[str] = []
    size = 0
    index: list[int] = []
    context: dict[str, str] = {}  # 直近の heading / header
    for i, (text, kind) in enumerate(lines):
        if current and (size + len(text) > MAX_CHARS or (kind == "heading" and size > MAX_CHARS / 2)):
            chunks.append(("\n".join(current), index))
            current = [c for c in (context.get("heading"), context.get("header")) if c and c != text]
            size = sum(len(c) + 1 for c in current)
            index = []
        current.append(text)
        index.append(i)
        size += len(text) + 1
        if kind:
            context[kind] = text
    if current:
        chunks.append(("\n".join(current), index))
    return chunks


def chunk_sections(sections: list[Section]) -> list[Chunked]:
    """位置単位ごとに split_chunks で切る。"""
    return [(name, split_chunks(lines)) for name, lines in sections]
