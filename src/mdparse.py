"""Lectura del Markdown generado: front-matter + bloques con página y sección."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from .common import SECCIONES
from .extract import detect_sections

PAGE_RE = re.compile(r"^<!-- pagina: (\d+) -->$")


def read_md(path: Path):
    text = Path(path).read_text(encoding="utf-8")
    meta = {}
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        meta = yaml.safe_load(fm) or {}
    else:
        body = text
    return meta, body


def lines_with_page(body: str):
    page, out = 0, []
    for line in body.splitlines():
        m = PAGE_RE.match(line.strip())
        if m:
            page = int(m.group(1))
            out.append(("", page))  # la marca actúa como separador de bloque
            continue
        out.append((line, page))
    return out


def blocks(body: str):
    """Bloques = párrafos, tablas completas o imágenes; cada uno con página y sección.

    Una tabla que continúa en la página siguiente se mantiene como bloques separados
    por página para no perder la trazabilidad de página.
    """
    lines = lines_with_page(body)
    found = detect_sections(lines)
    starts = sorted((v["linea"], v["numero_en_documento"], k) for k, v in found.items())

    def section_at(i):
        sec = (0, "Encabezado / sin sección")
        for li, num_doc, canon in starts:
            if li <= i:
                sec = (num_doc, SECCIONES[canon])
        return sec

    out, cur, cur_kind = [], [], None

    def flush(end_i):
        nonlocal cur, cur_kind
        if cur:
            txt = "\n".join(l for l, _, _ in cur).strip()
            if txt:
                first_i = cur[0][2]
                num, name = section_at(first_i)
                out.append({"text": txt, "page": cur[0][1], "page_end": cur[-1][1], "section": num,
                            "section_name": name, "kind": cur_kind, "line": first_i})
        cur, cur_kind = [], None

    for i, (line, page) in enumerate(lines):
        s = line.strip()
        is_table = s.startswith("|")
        is_heading = (i in {v["linea"] for v in found.values()}) or s.startswith("#")
        kind = "table" if is_table else ("image" if s.startswith("![") or s.startswith("> Imagen") else "text")
        if not s:
            if cur_kind != "table" or True:
                flush(i)
            continue
        if cur and (cur[-1][1] != page or is_heading or (cur_kind == "table") != is_table):
            flush(i)
        if is_heading:
            cur_kind = "heading"
            cur.append((line, page, i))
            flush(i)
            continue
        if not cur:
            cur_kind = kind
        cur.append((line, page, i))
    flush(len(lines))
    # Unir el pie de imagen ("> Imagen ...") con su imagen
    merged = []
    for b in out:
        if merged and b["text"].startswith("> Imagen") and merged[-1]["kind"] == "image":
            merged[-1]["text"] += "\n" + b["text"]
        else:
            merged.append(b)
    return merged
