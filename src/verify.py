"""Verificación de la extracción contra el PDF original.

Para cada FDS:
  1. Compara, página por página, los "tokens críticos" (números, CAS, códigos H/P,
     unidades) del texto nativo del PDF con los del Markdown. Lo que falta o sobra
     se registra como posible error de extracción (errores_extraccion.csv).
  2. Elige al menos una tabla o bloque con datos (prioridad secciones 3, 9, 8),
     calcula su cobertura de tokens contra la página del PDF y renderiza la
     página a PNG para la comparación visual manual (verificacion_bloques.csv,
     verificacion/<doc_id>_pN.png, verificacion.md).

La columna resultado_manual / errores_observados la llena el grupo tras mirar
el PNG: la verificación automática NO reemplaza la revisión visual.

Uso: python -m src.verify
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import pymupdf

from .common import ROOT, load_config
from .mdparse import blocks, read_md

TOKEN_RE = re.compile(
    r"\b\d{2,7}-\d{2}-\d\b"                      # N.º CAS
    r"|\b[HP]\d{3}(?:\+[HP]\d{3})*\b"            # frases H / P
    r"|\bUN\s?\d{4}\b"                           # número ONU
    r"|-?\d+(?:[.,]\d+)?"                        # números
    r"|°C|ppm|mg/m3|mg/m³|g/cm3|g/cm³|g/L|g/l|mg/kg|mg/L|kPa|mmHg|%",
    re.IGNORECASE,
)
PRIORITY = [3, 9, 8, 2, 14, 11]


def tokens(text: str) -> Counter:
    t = text.replace(" ", " ")
    t = re.sub(r"<!--.*?-->", " ", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)        # rutas de imágenes no cuentan
    t = re.sub(r"> Imagen \(pág\. \d+\)\.", " ", t)      # nota añadida por el pipeline
    return Counter(m.group(0).upper().replace(" ", "") for m in TOKEN_RE.finditer(t))


def page_md_text(body: str) -> dict[int, str]:
    pages, cur = {}, 0
    for line in body.splitlines():
        m = re.match(r"^<!-- pagina: (\d+) -->$", line.strip())
        if m:
            cur = int(m.group(1))
            continue
        pages.setdefault(cur, []).append(line)
    return {p: "\n".join(v) for p, v in pages.items()}


def pick_blocks(bl):
    tables = [b for b in bl if b["kind"] == "table"]
    for sec in PRIORITY:
        cand = [b for b in tables if b["section"] == sec]
        if cand:
            return cand[0], "tabla"
    if tables:
        return tables[0], "tabla"
    dense = sorted((b for b in bl if b["kind"] == "text"), key=lambda b: -sum(tokens(b["text"]).values()))
    return (dense[0], "bloque_con_datos") if dense else (None, "")


def main():
    cfg = load_config()
    md_root = Path(cfg["paths"]["md_dir"])
    pdf_dir = Path(cfg["paths"]["pdf_dir"])
    rep = Path(cfg["paths"]["reports_dir"])
    img_out = rep / "verificacion"
    img_out.mkdir(parents=True, exist_ok=True)

    err_rows, ver_rows, md_report = [], [], ["# Verificación de extracción (evidencia)\n"]
    for d in sorted(p for p in md_root.iterdir() if p.is_dir()):
        mdp = d / f"{d.name}.md"
        if not mdp.exists():
            continue
        meta, body = read_md(mdp)
        raw_pages = json.loads((d / "raw_pages.json").read_text(encoding="utf-8"))
        scanned = set(meta.get("paginas_escaneadas") or [])
        md_pages = page_md_text(body)

        # 1) Comparación por página
        for p, raw in enumerate(raw_pages, start=1):
            tr, tm = tokens(raw), tokens(md_pages.get(p, ""))
            falt, sobr = tr - tm, tm - tr
            if falt or sobr:
                err_rows.append({
                    "doc_id": d.name, "pagina": p, "escaneada": p in scanned,
                    "tokens_pdf": sum(tr.values()), "tokens_md": sum(tm.values()),
                    "faltan_en_md": " ".join(sorted(falt.elements()))[:500],
                    "sobran_en_md": " ".join(sorted(sobr.elements()))[:500],
                    "interpretacion": ("página escaneada: el PDF no tiene texto nativo, comparar con la imagen"
                                       if p in scanned else
                                       "posible pérdida/alteración en la conversión; revisar" if falt else
                                       "tokens extra (OCR de imágenes o duplicados de encabezado); revisar"),
                })

        # 2) Bloque/tabla para verificación visual
        bl = blocks(body)
        b, kind = pick_blocks(bl)
        if not b:
            continue
        page = b["page"]
        tb, tp = tokens(b["text"]), tokens(raw_pages[page - 1] if page else "")
        found = sum(min(c, tp[k]) for k, c in tb.items())
        total = sum(tb.values()) or 1
        png = img_out / f"{d.name}_p{page}.png"
        with pymupdf.open(pdf_dir / meta["archivo"]) as doc:
            doc[page - 1].get_pixmap(dpi=110).save(png)
        recall = round(found / total, 3)
        ver_rows.append({
            "doc_id": d.name, "fabricante": meta.get("fabricante"), "producto": meta.get("producto"),
            "pagina": page, "seccion": b["section"], "tipo_bloque": kind,
            "tokens_bloque_md": total, "tokens_encontrados_en_pdf": found, "cobertura_auto": recall,
            "estado_auto": "OK" if recall >= 0.98 else "REVISAR",
            "faltan_en_pdf": " ".join(sorted((tb - tp).elements()))[:300],
            "imagen_pagina": str(png.relative_to(rep)),
            "resultado_manual": "", "errores_observados": "", "revisado_por": "",
        })
        md_report += [f"## {meta.get('fabricante')} — {meta.get('producto')} (`{d.name}`)\n",
                      f"Página {page}, sección {b['section']}, {kind}. Cobertura automática de tokens: **{recall:.0%}**. ",
                      f"Imagen de la página: `{png.relative_to(rep)}`\n",
                      "**Markdown extraído:**\n", b["text"] if kind == "tabla" else "> " + b["text"].replace("\n", "\n> "),
                      "\n**Resultado de la revisión visual:** _(completar: OK / errores encontrados)_\n"]

    with open(rep / "errores_extraccion.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["doc_id", "pagina", "escaneada", "tokens_pdf", "tokens_md", "faltan_en_md", "sobran_en_md", "interpretacion"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(err_rows)
    with open(rep / "verificacion_bloques.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ver_rows[0]) if ver_rows else ["doc_id"])
        w.writeheader()
        w.writerows(ver_rows)
    (rep / "verificacion.md").write_text("\n".join(md_report), encoding="utf-8")
    n_rev = sum(1 for r in ver_rows if r["estado_auto"] != "OK")
    print(f"Bloques verificados: {len(ver_rows)} ({n_rev} marcados REVISAR). Páginas con diferencias: {len(err_rows)}")
    print(f"Ver {(rep / 'verificacion_bloques.csv').relative_to(ROOT)}, errores_extraccion.csv y verificacion.md")


if __name__ == "__main__":
    main()
