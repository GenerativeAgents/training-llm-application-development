"""位置単位(シート / スライド / ページ)の行を、埋め込み用のチャンクに切る共通部品。

どう切るかは抽出器の責任で、各抽出器の extract() が最後にここを呼ぶ(自前の切り方に替えてもよい)。
チャンクの ID(`#位置単位#連番`)は ingest.py が付けるので、ここでは本文だけを返す。
"""

from . import Chunked, Line, Section

MAX_CHARS = 3000  # 1 文書の目安。text-embedding-3-small の上限(8191 トークン)に十分収まる
# 残りがこの長さ以内なら、MAX_CHARS を超えても切らずに今のチャンクに入れる(見出しと数行だけの小さな末尾チャンクを作らない。
# 小さなチャンクは本文の無い題だけで検索に当たり、本体のチャンクを押しのける)。チャンクの上限は MAX_CHARS + TAIL_SLACK
TAIL_SLACK = 300


def split_chunks(lines: list[Line]) -> list[str]:
    """MAX_CHARS を目安に分割する。半分を超えていれば見出し行で切る。残りが TAIL_SLACK 以内なら切らない。

    2 つ目以降のチャンクには、直前の見出し行と表のヘッダ行を先頭に付け直す(節の途中から始まるチャンクに文脈を与える)。
    ただし見出し行で切ったときは、そこから新しい節なので、前の節の見出しも表のヘッダも付け直さない
    (付けると、前の節の語で検索に当たり、回答時にも節の対応を取り違える)。
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
    rest = [0] * (len(lines) + 1)  # rest[i]: i 行目以降の残りの文字数
    for i in range(len(lines) - 1, -1, -1):
        rest[i] = rest[i + 1] + len(lines[i][0]) + 1
    for i, (text, kind) in enumerate(lines):
        # 文字数で切るのは、残り全部を入れても MAX_CHARS + TAIL_SLACK に収まらないときだけ
        over = size + len(text) > MAX_CHARS and size + rest[i] > MAX_CHARS + TAIL_SLACK
        if current and (over or (kind == "heading" and size > MAX_CHARS / 2)):
            chunks.append(("\n".join(current), index))
            # 見出しで切ったなら、この行から新しい節なので、前の節の見出しも表のヘッダも付け直さない
            carry = () if kind == "heading" else (context.get("heading"), context.get("header"))
            current = [c for c in carry if c and c != text]
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
