"""段階 4(vision 抽出器)が xlsx の 1 シートをどう処理するかを、LLM を呼ばずにキャッシュから復元して書き出す。

既定は画面遷移図のシート `2. プロジェクト管理(A102)`。別のファイル・シートで作るなら引数で指定する。
キャッシュ(data/cache/vision/)に無い組み合わせは LLM の出力を書き出さない(呼ばない)。
LibreOffice(soffice)があれば、比較用に LibreOffice でシートを PDF にしてページを PNG にしたものも書き出す
(自前で描く前に試して、コネクタの先がつながらなかった描き方)。

    uv run python docs/stage4_vision_sample/make_sample.py
    uv run python docs/stage4_vision_sample/make_sample.py --file <corpus からの相対パス.xlsx> --sheet <シート名> --out <出力先>
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.advanced_rag.extractors import structured, vision  # noqa: E402
from app.advanced_rag.extractors.arrows import prompt_lines  # noqa: E402
from app.advanced_rag.extractors.render_drawing import render_sheets  # noqa: E402
from app.advanced_rag.ingest import CORPUS_DIR, ingest_file  # noqa: E402
from app.advanced_rag.rag import docs_path  # noqa: E402
from app.advanced_rag.vision_config import VISION_MODEL, VISION_REASONING_EFFORT  # noqa: E402

DEFAULT_FILE = "システム開発/A1_プロジェクト管理システム/010_要件定義/040_画面設計/画面遷移図_A1_プロジェクト管理システム.xlsx"
DEFAULT_SHEET = "2. プロジェクト管理(A102)"


def libreoffice_pages(path: Path, sheet: str) -> list[bytes]:
    """LibreOffice でブックを PDF にし、sheet の印刷ページを PNG(2 倍)にして返す。soffice が無ければ空。

    ページの特定は本文の見出し(シート名と同じ文字)が載る最後のページ(目次にも同じ文字が載るので先頭は取らない)で、
    続きのページはフッタの `- 2 -`, `- 3 -` … で追う(このブックの印刷設定向けの目安)。
    """
    if not shutil.which("soffice"):
        print("soffice が無いので LibreOffice の描画は飛ばす")
        return []
    import pypdfium2 as pdfium

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["soffice", f"-env:UserInstallation=file://{tmp}/profile", "--headless", "--convert-to", "pdf",
                        "--outdir", tmp, str(path)], check=True, capture_output=True, timeout=600)
        pdf = pdfium.PdfDocument(Path(tmp) / f"{path.stem}.pdf")
        texts = [re.sub(r"\s", "", pdf[i].get_textpage().get_text_range()) for i in range(len(pdf))]
        hits = [i for i, t in enumerate(texts) if re.sub(r"\s", "", sheet) in t]
        if not hits:
            return []
        start = hits[-1]
        pages, n = [start], 2
        while start + n - 1 < len(pdf) and f"-{n}-" in texts[start + n - 1]:
            pages.append(start + n - 1)
            n += 1
        out = []
        for i in pages:
            im = pdf[i].render(scale=2).to_pil()
            buf = __import__("io").BytesIO()
            im.save(buf, format="PNG")
            out.append(buf.getvalue())
        return out


def main(rel: str, sheet: str, out: Path) -> None:
    path = CORPUS_DIR / rel
    out.mkdir(parents=True, exist_ok=True)
    write = lambda name, text: (out / name).write_text(text, encoding="utf-8")  # noqa: E731

    # (1) 段階 3(structured)の行: セルの文字 + 図形テキスト(出現順)。vision はこの後ろに「図の説明:」行を足す
    base = dict(structured.sections(path))
    write("01_structured_lines.txt", "\n".join(text for text, _ in base[sheet]))

    # (2) LLM に渡す画像: シート(セル・図形・画像)を render_drawing で PNG に描いたもの
    png = vision.png_bytes(render_sheets(path, only=sheet)[sheet])
    (out / "02_sheet_rendered.png").write_bytes(png)

    # (2b) 比較用: LibreOffice に描かせたもの(採用しなかった描き方)
    for old in out.glob("02b_libreoffice_page*.png"):
        old.unlink()
    for i, png_lo in enumerate(libreoffice_pages(path, sheet), 1):
        (out / f"02b_libreoffice_page{i}.png").write_bytes(png_lo)

    # (3) drawing XML から確定した矢印の一覧(プロンプトの末尾に添える)
    arrows = vision.sheet_arrows(path, {sheet})[sheet]
    extra = prompt_lines(arrows)
    write("03_arrows_from_xml.txt", extra.strip())

    # (4) プロンプト全文 = PROMPT(figure_rules の RULES / OMIT / SCHEMA を含む) + 矢印の一覧
    prompt = vision.PROMPT + extra
    write("04_prompt_full.txt", prompt)

    # (5) LLM の出力(Figure の JSON)。describe() と同じキーでキャッシュを引く
    file_hash = hashlib.sha1(path.read_bytes()).hexdigest()
    image_hash = hashlib.sha1(png).hexdigest()
    key = hashlib.sha1(f"{file_hash}\n{sheet}\n{image_hash}\n{VISION_MODEL}\n{VISION_REASONING_EFFORT}\nFigure\n{prompt}".encode()).hexdigest()
    cache = vision.CACHE_DIR / f"{key}.txt"
    print(f"model={VISION_MODEL} effort={VISION_REASONING_EFFORT} cache={cache.name} exists={cache.exists()}")
    if cache.exists():
        raw = cache.read_text(encoding="utf-8")
        write("05_llm_output.json", json.dumps(json.loads(raw), ensure_ascii=False, indent=2))
        # (6) figure_lines がコードで「図の説明:」行に組み立てる(矢印の向きは XML の一覧で直す)
        write("06_figure_lines.txt", "\n".join(line for _, line in vision.figure_lines(raw, arrows)))

    # (7) ingest が ID・メタ情報を付けたレコード(docs_vision.jsonl に書かれる形)。
    # 配布済みの docs_vision.jsonl に同じ ID があればそれも書き、違いがあれば知らせる
    doc_id = f"{Path(rel).with_suffix('').as_posix()}#{sheet}"
    built = [r for r in ingest_file(path, "vision") if r["id"] == doc_id or r["id"].startswith(doc_id + "#")]
    write("07_db_record.json", json.dumps(built, ensure_ascii=False, indent=2))
    jsonl = docs_path("vision")
    if jsonl.exists():
        shipped = [json.loads(l) for l in jsonl.open(encoding="utf-8")]
        shipped = [r for r in shipped if r["id"] == doc_id or r["id"].startswith(doc_id + "#")]
        if shipped and shipped != built:
            write("07_db_record_shipped.json", json.dumps(shipped, ensure_ascii=False, indent=2))
            print(f"注意: {jsonl.name} のレコードは現行コードの結果と違う(作成時のプロンプトか描画が違う)。07_db_record_shipped.json に書いた")
    print(f"-> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=DEFAULT_FILE)
    p.add_argument("--sheet", default=DEFAULT_SHEET)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent))
    a = p.parse_args()
    main(a.file, a.sheet, Path(a.out))
