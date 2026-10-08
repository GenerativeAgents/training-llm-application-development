"""全文検索(FTS)用の区切りと問い合わせ。

日本語は空白で語が分かれないので、LanceDB の FTS に渡す前に自前で区切り、空白区切りの列にしておく
(LanceDB 側は空白で分けるだけ)。実務でよくあるように、形態素と bi-gram の 2 通りで区切って併用する。

- 形態素(`fts_morph`): Sudachi のモード C(長い単位。「死亡率」「交付金」)。日本語の語は表記ゆれを
  正規化する(フィルタ → フィルター、引落し → 引き落とし)。英数字の語は正規化すると壊れる
  (project → プロジェクト)ので、表層を小文字にするだけ。辞書の区切りと検索語の区切りが合えば精度が高い
- bi-gram(`fts_bigram`): かな漢字の並びを 2 文字ずつ重ねて区切り、英数字の語はそのまま 1 語にする
  (Lucene の CJK bi-gram と同じ)。辞書に頼らないので、複合語の一部や ID の部分でも当たる
- 1 文字(`fts_unigram`): かな漢字を 1 文字ずつ。bi-gram では 1 文字の検索語(「胃」)が引けないので、その受け皿

LanceDB 標準の `ngram` トークナイザは使わない。検索語の n-gram をばらばらに OR で探すだけで、並びの一致
(フレーズ)が効かず、`FB1999904` が `19` を含むだけのチャンクにも当たる。

問い合わせは空白区切りの語(質問文から品詞で抜き出したキーワード)で、Elasticsearch の
match + match_phrase と同じ形にする。語を区切った単位の OR で広く拾い(BM25)、語の並びがそのまま
一致したチャンクには、語ごとのフレーズ一致の点を足す。フレーズ一致だけにすると、質問の複合語が文書に
そのままの形で無いとき(質問は「顧客検索API」、文書は「顧客の検索API」)に取りこぼす。
"""

import re
import threading
import unicodedata

from lancedb.query import BooleanQuery, FullTextQuery, MatchQuery, Occur, PhraseQuery
from sudachipy import Dictionary, SplitMode
from sudachipy.morpheme import Morpheme
from sudachipy.tokenizer import Tokenizer

COLUMNS = {"morph": "fts_morph", "bigram": "fts_bigram"}  # 検索に使う列(RagModel.fts_fields で選ぶ)
UNIGRAM = "fts_unigram"
ALL_COLUMNS = [*COLUMNS.values(), UNIGRAM]

_CJK = "぀-ヿ㐀-鿿豈-﫿々〆ー"
WORD_CHAR = re.compile(rf"[0-9a-zA-Z{_CJK}]")
# 英数字の語(記号でつながった ID・パスは 1 語: project_name、Ctrl+Shift+S は ctrl / shift / s)と、かな漢字の並び
RUN = re.compile(rf"[0-9a-z]+(?:[._\-/][0-9a-z]+)*|[{_CJK}]+")
SUDACHI_MAX = 4000  # Sudachi の 1 回の入力の上限(バイト数)より十分短く切る

_sudachi = threading.local()  # Sudachi の Tokenizer は同時に使えない(Weave の評価は predict を並列に呼ぶ)ので、スレッドごとに作る


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def morphs(text: str) -> list[str]:
    out = []
    for line in unicodedata.normalize("NFKC", text).splitlines():
        for i in range(0, len(line), SUDACHI_MAX):
            for m in _tokenizer().tokenize(line[i : i + SUDACHI_MAX]):
                word = m.surface()
                if not WORD_CHAR.search(word):  # 記号・空白は語にしない
                    continue
                out.append(word.lower() if word.isascii() else m.normalized_form().lower())
    return out


def _tokenizer() -> Tokenizer:
    if not hasattr(_sudachi, "tokenizer"):
        _sudachi.tokenizer = Dictionary(dict="core").tokenizer(SplitMode.C)
    return _sudachi.tokenizer


# 質問からキーワードを抜き出すときに残す品詞(Sudachi の品詞の 1 段目)。名詞の前後に付く接頭辞・接尾辞も名詞の一部にする
CONTENT_POS = {"名詞", "接頭辞", "接尾辞"}
ID_JOINERS = set("_-./")  # 英数字の語の間にあれば、識別子の一部として続ける(project_name、e-Gov)
QUESTION_NOUNS = {"いくつ", "いくら", "いつ"}  # 普通名詞として出てくる疑問の語


def keywords(question: str) -> str:
    """質問文から全文検索用のキーワード(空白区切り)を、形態素の品詞で抜き出す(LLM を使わない)。

    名詞(と、それに付く接頭辞・接尾辞)が続いている部分を、元の表記のまま 1 語にする(「主キーカラム」「FB1999904」)。
    助詞・助動詞・動詞・代名詞・記号は落とす。「場合」「とき」「全て」のような副詞的な名詞は、単独なら落とし、
    ほかの名詞と続いていれば残す(「1960年ごろ」)。空白・記号(英数字の間の _ - . / を除く)で語を切る。
    """
    text = unicodedata.normalize("NFKC", question)
    words: list[str] = []
    run: list[Morpheme] = []  # run: 続いている内容語の形態素
    def flush() -> None:
        if run and not (len(run) == 1 and ("副詞可能" in run[0].part_of_speech() or run[0].surface() in QUESTION_NOUNS)):
            words.append(text[run[0].begin() : run[-1].end()])
        run.clear()
    tokens = list(_tokenizer().tokenize(text))
    for i, m in enumerate(tokens):
        pos = m.part_of_speech()[0]
        if pos in CONTENT_POS:
            if run and run[-1].end() != m.begin():
                flush()
            run.append(m)
        elif (m.surface() in ID_JOINERS and run and run[-1].surface().isascii() and i + 1 < len(tokens)
              and tokens[i + 1].surface().isascii() and tokens[i + 1].part_of_speech()[0] in CONTENT_POS):
            run.append(m)
        else:
            flush()
    flush()
    return " ".join(dict.fromkeys(words))  # 同じ語は 1 回


def bigrams(text: str) -> list[str]:
    out = []
    for run in RUN.findall(normalize(text)):
        if run.isascii() or len(run) == 1:
            out.append(run)
        else:
            out += [run[i : i + 2] for i in range(len(run) - 1)]
    return out


def unigrams(text: str) -> list[str]:
    return [c for c in normalize(text) if WORD_CHAR.match(c) and not c.isascii()]


def columns(text: str) -> dict[str, str]:
    """文書 1 件分の FTS 用の列(空白区切りのトークン)。"""
    return {
        COLUMNS["morph"]: " ".join(morphs(text)),
        COLUMNS["bigram"]: " ".join(bigrams(text)),
        UNIGRAM: " ".join(unigrams(text)),
    }


def _field_queries(column: str, text: str) -> list[FullTextQuery]:
    tokenize = morphs if column == COLUMNS["morph"] else bigrams
    tokens = tokenize(text)
    queries: list[FullTextQuery] = []
    if column == COLUMNS["bigram"]:
        # かな漢字 1 文字の語は bi-gram の列に無いので、1 文字の列で引く
        singles = [t for t in tokens if len(t) == 1 and not t.isascii()]
        tokens = [t for t in tokens if t not in singles]
        if singles:
            queries.append(MatchQuery(" ".join(singles), UNIGRAM))
    if tokens:
        queries.append(MatchQuery(" ".join(tokens), column))  # 語の OR
    for word in text.split():  # 語ごとのフレーズ一致を加点
        phrase = tokenize(word)
        if len(phrase) >= 2:
            queries.append(PhraseQuery(" ".join(phrase), column))
    return queries


def query(text: str, fields: list[str]) -> BooleanQuery | None:
    """空白区切りの語 text の問い合わせ。fields は COLUMNS のキー。各列の点を足し合わせる。"""
    parts = [q for f in fields for q in _field_queries(COLUMNS[f], text)]
    return BooleanQuery([(Occur.SHOULD, q) for q in parts]) if parts else None
