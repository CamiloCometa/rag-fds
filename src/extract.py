"""Extracción documental: PDF -> Markdown con trazabilidad de página, sección e imágenes.

Uso:
    python -m src.extract                # procesa todo data/pdfs/
    python -m src.extract --only ColorSur

Salidas por FDS (output/md/<doc_id>/):
    <doc_id>.md        Markdown con front-matter, marcas <!-- pagina: N --> y rutas relativas a images/
    images/            imágenes con información técnica (pictogramas, logos, tablas-imagen…)
    images.csv         página, sección, bbox, OCR, tabla asociada (solo con evidencia geométrica)
    sections.json      estado de cada una de las 16 secciones
    raw_pages.json     texto plano de cada página según PyMuPDF (para verificación)
Salidas globales: data/manifest.csv, output/reports/extraction_report.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from pathlib import Path

import pymupdf
import pymupdf4llm

from .common import (ROOT, SECCIONES, load_config, match_heading, norm, select_sections, sha256,
                     slug)

MANIFEST_COLS = ["doc_id", "archivo", "carpeta", "fabricante", "marca", "producto", "version", "fecha_revision",
                 "n_paginas", "sha256", "procedencia", "fecha_recepcion", "campos_auto", "notas"]
MANUAL_COLS = {"procedencia", "fecha_recepcion", "notas"}

PRODUCT_RES = [
    r"nombre\s+(?:comercial\s+)?del\s+producto\s*[:：]\s*(?P<v>[^\n|]{3,120})",
    r"nombre\s+comercial\s*[:：]\s*(?P<v>[^\n|]{3,120})",
    r"identificador\s+del\s+producto\s*[:：]?\s*(?:nombre\s*[:：]\s*)?(?P<v>[^\n|]{3,120})",
    r"product\s+name\s*[:：]\s*(?P<v>[^\n|]{3,120})",
    r"nombre\s*[:：]\s*(?P<v>[^\n|]{3,120})",
]
VERSION_RE = r"versi[oó]n\s*(?:n[º°o.]*\s*)?[:：]?\s*(?P<v>\d+(?:[.,]\d+)*)"
DATE_RE = (r"(?:fecha\s+de\s+(?:revisi[oó]n|actualizaci[oó]n|emisi[oó]n|elaboraci[oó]n|impresi[oó]n)|revisi[oó]n|revisado)"
           r"\s*[:：]?\s*(?P<v>\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\d{4}[/.\-]\d{2}[/.\-]\d{2}|\d{1,2}\s+de\s+\w+\s+de\s+\d{4})")
IMG_LINK_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")


FAB_LABELS = [  # en orden de especificidad
    r"fabricante\s*/\s*distribuidor",
    r"datos\s+del\s+proveedor\s+o\s+fabricante",
    r"(?:datos|informaci[oó]n)\s+(?:sobre\s+el|del)\s+proveedor(?:\s+de\s+la\s+ficha\s+de\s+datos\s+de\s+seguridad)?",
    r"nombre\s+de\s+la\s+(?:empresa|compa[ñn][ií]a)",
    r"raz[oó]n\s+social",
    r"empresa",
    r"compa[ñn][ií]a",
    r"fabricante",
    r"proveedor",
    r"manufacturer",
    r"supplier",
    r"distribuidor",
]
FAB_STOP = re.compile(r"\s*(?:,|;|\s-\s|\bNIT\b|\bNit\b|\bDirecci[oó]n\b|\bTel[eé]?f?|\bTfno|\bPbx|\bCalle\b|\bCra\b|"
                      r"\bCarrera\b|\bAv(?:enida)?\b|\bKm\b|\bwww\.|\bE-?mail\b|\bCorreo\b|\s{3,})", re.IGNORECASE)
FAB_BAD = re.compile(r"(?i)^(?:fabricante|distribuidor|proveedor|empresa|datos|de\s+la|del\s|nit|direcci|tel|y\s+de|"
                     r"identificaci|informaci|uso|nombre|c[oó]digo|n[uú]mero|para|la\s|el\s|o\s|y\s|en\s|persona|responsable|"
                     r"responsabilidad|direcci|tel[eé]fono|ficha|hoja|\d)")


def find_fabricante(text: str) -> str:
    """Busca la etiqueta Fabricante/Proveedor/Empresa en las primeras páginas y toma el valor que sigue
    (en la misma línea o en las siguientes). Devuelve '' si no hay un valor plausible."""
    for lab in FAB_LABELS:
        for m in re.finditer(rf"(?i)\b{lab}\b", text):
            rest = text[m.end():m.end() + 300]
            rest = re.sub(r"^[\s:：.*|]+", "", rest)
            for cand in rest.splitlines()[:3]:
                val = FAB_STOP.split(cand.strip(" :.-*|"))[0].strip(" :.-*|")
                if not val:
                    continue
                if 3 <= len(val) <= 80 and re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]{3,}", val) and not FAB_BAD.match(val):
                    return val
                break  # primera línea con texto no plausible: probar la siguiente aparición
    return ""


def from_filename(stem: str, text: str):
    """Nombres del tipo 'FDS 45 - VINILTEX PRO 650 BLANCO - PINTUCO'.

    Devuelve (producto, fabricante). El fabricante (último segmento) solo se acepta si
    tiene 1-2 palabras y aparece en el texto del PDF (evidencia de que es el fabricante).
    """
    parts = [x.strip(" .-_") for x in re.split(r"\s+-\s*|\s*-\s+", stem.strip()) if x.strip(" .-_")]
    if parts and re.fullmatch(r"(?i)fds\s*\d+", parts[0]):
        parts = parts[1:]
    fab = ""
    last = parts[-1] if parts else ""
    if len(parts) >= 2 and 1 <= len(last.split()) <= 2 and last.isupper() and len(last) >= 3:
        fab = last
        parts = parts[:-1]
        if len(parts) >= 2 and parts[-1].isupper() and len(parts[-1].split()) == 1:  # "... - INTERNATIONAL - AKSONOBEL"
            fab = f"{parts[-1]} {fab}"
            parts = parts[:-1]
    prod = " - ".join(parts)
    if " " not in prod and prod.count("-") >= 2:  # "pintura-antideslizante-para-canchas"
        prod = prod.replace("-", " ")
    return prod, fab


FRONT_KEYS = ("doc_id", "archivo", "carpeta", "fabricante", "marca", "producto", "version", "fecha_revision",
              "n_paginas", "sha256")


def front_matter(meta: dict, scanned) -> str:
    lines = ["---"] + [f"{k}: {json.dumps(str(meta.get(k, '') or ''), ensure_ascii=False)}" for k in FRONT_KEYS]
    return "\n".join(lines + ["paginas_escaneadas: " + json.dumps(list(scanned)), "---", ""]) + "\n"


def auto_metadata(stem: str, raw_pages: list[str]) -> dict:
    head = "\n".join(raw_pages[:2])
    prod_fn, marca = from_filename(stem, "\n".join(raw_pages))
    return {"fabricante": find_fabricante(head) or marca, "marca": marca,
            "producto": prod_fn or find_first(PRODUCT_RES, head), "version": find_first(VERSION_RE, head),
            "fecha_revision": find_first(DATE_RE, "\n".join(raw_pages))}


def refresh_metadata(cfg):
    """Vuelve a detectar fabricante/marca/producto/versión desde raw_pages.json (sin convertir PDF).
    Solo reemplaza campos que habían sido llenados automáticamente; respeta lo corregido a mano."""
    from .mdparse import read_md
    man_path = Path(cfg["paths"]["manifest"])
    manifest = load_manifest(man_path)
    by_id = {r["doc_id"]: r for r in manifest.values()}
    md_root = Path(cfg["paths"]["md_dir"])
    n = 0
    for d in sorted(p for p in md_root.iterdir() if p.is_dir()):
        mdp = d / f"{d.name}.md"
        row = by_id.get(d.name)
        if not mdp.exists() or not row:
            continue
        raw_pages = json.loads((d / "raw_pages.json").read_text(encoding="utf-8"))
        auto = auto_metadata(Path(row["archivo"]).stem, raw_pages)
        was_auto = set((row.get("campos_auto") or "").split(",")) - {""}
        filled = set(was_auto)
        for k, v in auto.items():
            cur = row.get(k, "")
            if (k in was_auto or not cur or cur == "SIN_IDENTIFICAR") and v:
                row[k] = v
                filled.add(k)
        row["fabricante"] = row.get("fabricante") or "SIN_IDENTIFICAR"
        row["campos_auto"] = ",".join(sorted(filled))
        meta, body = read_md(mdp)
        meta.update({k: row.get(k, "") for k in FRONT_KEYS if k in row})
        mdp.write_text(front_matter(meta, meta.get("paginas_escaneadas") or []) + body.lstrip("\n"), encoding="utf-8")
        mj = d / "meta.json"
        if mj.exists():
            m2 = json.loads(mj.read_text(encoding="utf-8"))
            m2.update(row)
            mj.write_text(json.dumps(m2, ensure_ascii=False, indent=1), encoding="utf-8")
        n += 1
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(manifest.values(), key=lambda r: r["archivo"]))
    # actualizar fabricante/producto en el reporte de extracción
    rep_path = Path(cfg["paths"]["reports_dir"]) / "extraction_report.csv"
    if rep_path.exists():
        rows = list(csv.DictReader(open(rep_path, encoding="utf-8")))
        for r in rows:
            if r.get("doc_id") in by_id:
                r["fabricante"] = by_id[r["doc_id"]]["fabricante"]
                r["producto"] = by_id[r["doc_id"]]["producto"]
        with open(rep_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return n


def find_first(patterns, text):
    for p in ([patterns] if isinstance(patterns, str) else patterns):
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return re.sub(r"[*_#]+", "", m.group("v")).strip(" .:-")
    return ""


def detect_sections(md_lines_with_page):
    """Encabezados de sección del Markdown → {sección canónica: {...}} (ver common.select_sections)."""
    cands = []
    for i, (line, page) in enumerate(md_lines_with_page):
        h = match_heading(line)
        if h:
            cands.append((i, h[0], h[2], h[1], page))
    return select_sections(cands)


def raw_section_hits(raw_pages):
    """Secciones que aparecen en el texto plano del PDF (sin pasar por el conversor)."""
    cands, k = [], 0
    for p, txt in enumerate(raw_pages, start=1):
        for line in txt.splitlines():
            h = match_heading(line)
            if h:
                cands.append((k, h[0], h[2], line.strip()[:80], p))
            k += 1
    return {c: (v["pagina"], v["titulo_en_documento"]) for c, v in select_sections(cands).items()}


def section_states(body_lines, raw_pages, scanned):
    """Estado de las 16 secciones: presente / fallo_extraccion / no_determinado / ausente_en_original_probable."""
    found = detect_sections(body_lines)
    raw_hits = raw_section_hits(raw_pages)
    sections = []
    for n in range(1, 17):
        if n in found:
            st = {**found[n], "estado": "presente"}
        elif n in raw_hits:
            st = {"num": n, "estado": "fallo_extraccion",
                  "detalle": f"El encabezado aparece en el texto del PDF (pág. {raw_hits[n][0]}: «{raw_hits[n][1]}») "
                             "pero no se reconoció en el Markdown"}
        elif scanned:
            st = {"num": n, "estado": "no_determinado",
                  "detalle": f"Páginas escaneadas {scanned}: verificar manualmente si la sección existe"}
        else:
            st = {"num": n, "estado": "ausente_en_original_probable",
                  "detalle": "No aparece ni en el Markdown ni en el texto del PDF; confirmar visualmente"}
        st["nombre_canonico"] = SECCIONES[n]
        sections.append(st)
    return sections


def recompute_sections(cfg):
    """Recalcula sections.json y el reporte SIN volver a convertir los PDF (segundos, no minutos)."""
    from .mdparse import read_md
    md_root = Path(cfg["paths"]["md_dir"])
    rep_path = Path(cfg["paths"]["reports_dir"]) / "extraction_report.csv"
    rows = {r["doc_id"]: r for r in csv.DictReader(open(rep_path, encoding="utf-8"))} if rep_path.exists() else {}
    diag = []
    for d in sorted(p for p in md_root.iterdir() if p.is_dir()):
        mdp = d / f"{d.name}.md"
        if not mdp.exists():
            continue
        meta, body = read_md(mdp)
        lines, page = [], 0
        for line in body.splitlines():
            m = re.match(r"^<!-- pagina: (\d+) -->$", line.strip())
            if m:
                page = int(m.group(1))
            lines.append((line, page))
        raw_pages = json.loads((d / "raw_pages.json").read_text(encoding="utf-8"))
        secs = section_states(lines, raw_pages, meta.get("paginas_escaneadas") or [])
        (d / "sections.json").write_text(json.dumps(secs, ensure_ascii=False, indent=1), encoding="utf-8")
        by = lambda st: [x["num"] for x in secs if x["estado"] == st]  # noqa: E731
        if d.name in rows:
            rows[d.name].update({"secciones_presentes": len(by("presente")),
                                 "ausentes_original_probable": by("ausente_en_original_probable"),
                                 "fallo_extraccion": by("fallo_extraccion"), "no_determinado": by("no_determinado")})
        for x in secs:
            if x["estado"] != "presente":
                diag.append({"doc_id": d.name, "seccion": x["num"], "estado": x["estado"], "detalle": x.get("detalle", "")})
    if rows:
        with open(rep_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(next(iter(rows.values()))))
            w.writeheader()
            w.writerows(rows.values())
    out = Path(cfg["paths"]["reports_dir"]) / "secciones_no_presentes.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "seccion", "estado", "detalle"])
        w.writeheader()
        w.writerows(diag)
    return rows, diag


def summarize(cfg):
    """Resumen global de la extracción (para el informe)."""
    from collections import Counter
    rep_path = Path(cfg["paths"]["reports_dir"]) / "extraction_report.csv"
    rows = list(csv.DictReader(open(rep_path, encoding="utf-8")))
    man = list(csv.DictReader(open(cfg["paths"]["manifest"], encoding="utf-8")))
    ok = [r for r in rows if not r.get("error")]
    errs = [r for r in rows if r.get("error")]
    est = Counter()
    for r in ok:
        for k in ("fallo_extraccion", "ausentes_original_probable", "no_determinado"):
            for n in json.loads(r.get(k) or "[]"):
                est[(k, n)] += 1
    print(f"Documentos: {len(rows)} (con error: {len(errs)}) | páginas: {sum(int(r['paginas']) for r in ok)} | "
          f"tablas: {sum(int(r['tablas_md']) for r in ok)} | imágenes: {sum(int(r['imagenes']) for r in ok)}")
    print(f"Con 16/16 secciones: {sum(1 for r in ok if int(r['secciones_presentes']) == 16)} | "
          f"media de secciones presentes: {sum(int(r['secciones_presentes']) for r in ok) / max(1, len(ok)):.1f}")
    print(f"Documentos con páginas escaneadas (OCR): {sum(1 for r in ok if r['paginas_escaneadas'] not in ('[]', ''))}")
    for k, lab in (("fallo_extraccion", "Fallos de extracción"), ("ausentes_original_probable", "Ausentes en original (probable)"),
                   ("no_determinado", "No determinadas")):
        items = sorted((n, c) for (kk, n), c in est.items() if kk == k)
        print(f"{lab} por sección: " + (", ".join(f"S{n}×{c}" for n, c in items) or "ninguno"))
    fabs = Counter(r["fabricante"] for r in man)
    print("Fabricantes: " + ", ".join(f"{f} ({c})" for f, c in fabs.most_common()))
    for r in errs:
        print(f"ERROR {r['archivo']}: {r['error'][:150]}")


def ocr_image(path: Path, lang: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(path)
        if min(img.size) < 60:  # amplía imágenes pequeñas para mejorar el OCR
            f = 60 / min(img.size)
            img = img.resize((int(img.width * f * 2), int(img.height * f * 2)))
        txt = pytesseract.image_to_string(img, lang=lang)
    except Exception as e:  # noqa: BLE001
        return f"[OCR no disponible: {e}]"
    txt = re.sub(r"\s+", " ", txt).strip()
    words = re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúñ0-9]{2,}", txt)
    return txt if len(words) >= 2 else ""


def intersects(a, b, tol=2):
    return not (a[2] < b[0] - tol or b[2] < a[0] - tol or a[3] < b[1] - tol or b[3] < a[1] - tol)


def process_pdf(pdf: Path, cfg: dict, manifest_row: dict) -> dict:
    ex = cfg["extraction"]
    carpeta = pdf.parent.name                     # p. ej. "FDS 2018": año/lote de la ficha
    doc_id = manifest_row.get("doc_id") or f"{slug(carpeta)}__{slug(pdf.stem)}"
    out = Path(cfg["paths"]["md_dir"]) / doc_id
    if out.exists():
        shutil.rmtree(out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True)
    t0 = time.time()

    doc = pymupdf.open(pdf)
    raw_pages = [pg.get_text() for pg in doc]
    scanned = [i + 1 for i, (pg, t) in enumerate(zip(doc, raw_pages))
               if len(t.strip()) < 50 and pg.get_images()]

    chunks = pymupdf4llm.to_markdown(
        doc, page_chunks=True, write_images=True, image_path=str(img_dir),
        dpi=ex["image_dpi"], use_ocr=ex["use_ocr"], ocr_language=ex["ocr_language"],
        header=ex["keep_headers_footers"], footer=ex["keep_headers_footers"], show_progress=False,
    )

    # ---- Ensamblar Markdown por página, normalizando rutas de imágenes a relativas
    images, body_lines = [], []
    for ch in chunks:
        page = ch["metadata"].get("page_number") or ch["metadata"].get("page", 0)
        boxes = ch.get("page_boxes", [])
        tables = [b["bbox"] for b in boxes if b.get("class") == "table"]
        text = ch["text"]

        def fix(m):
            fname = Path(m.group("path")).name
            fpath = img_dir / fname
            if not fpath.exists():
                return m.group(0)
            try:
                from PIL import Image
                w, h = Image.open(fpath).size
            except Exception:  # noqa: BLE001
                w = h = 0
            if min(w, h) < ex["min_image_side_px"]:
                fpath.unlink(missing_ok=True)
                return ""
            idx = re.search(r"-(\d+)\.\w+$", fname)
            bbox = None
            if idx and int(idx.group(1)) < len(boxes):
                bbox = boxes[int(idx.group(1))]["bbox"]
            assoc = ""
            if bbox:
                for k, tb in enumerate(tables, start=1):
                    if intersects(bbox, tb):
                        assoc = f"tabla_{k}_pag_{page} (evidencia: bbox de imagen dentro/solapado con la tabla)"
            ocr = ocr_image(fpath, ex["ocr_language"]) if ex["ocr_images"] else ""
            images.append({"archivo": f"images/{fname}", "pagina": page, "ancho_px": w, "alto_px": h,
                           "bbox_pdf": [round(v, 1) for v in bbox] if bbox else "",
                           "tabla_asociada": assoc, "ocr_texto": ocr})
            note = f"\n\n> Imagen (pág. {page}). "
            note += f"OCR automático, revisar: «{ocr}»" if ocr else "Sin texto legible por OCR; contenido gráfico no interpretado."
            return f"![{m.group('alt') or 'imagen pág. ' + str(page)}](images/{fname}){note}\n"

        text = IMG_LINK_RE.sub(fix, text)
        body_lines.append((f"<!-- pagina: {page} -->", page))
        for line in text.splitlines():
            body_lines.append((line, page))

    # ---- Secciones
    sections = section_states(body_lines, raw_pages, scanned)
    # sección de cada imagen = última sección cuyo encabezado precede a la línea de la imagen
    starts = sorted((s["linea"], s["num"]) for s in sections if s["estado"] == "presente")
    for img in images:
        line_idx = next((i for i, (l, _) in enumerate(body_lines) if img["archivo"] in l), None)
        prev = [n for li, n in starts if line_idx is not None and li <= line_idx]
        img["seccion"] = prev[-1] if prev else ""

    # ---- Metadatos (auto) — el manifiesto conserva lo que el grupo corrija a mano
    head = "\n".join(raw_pages[:2])
    auto = auto_metadata(pdf.stem, raw_pages)
    meta = dict(manifest_row)
    filled = []
    for k, v in auto.items():
        if not meta.get(k) and v:
            meta[k] = v
            filled.append(k)
    fabricante = meta.get("fabricante") or "SIN_IDENTIFICAR"
    meta.update({"doc_id": doc_id, "carpeta": carpeta, "fabricante": fabricante, "n_paginas": len(doc),
                 "sha256": sha256(pdf), "campos_auto": ",".join(filled) or meta.get("campos_auto", "")})
    if not meta.get("producto"):
        meta["producto"] = pdf.stem

    md_text = front_matter(meta, scanned) + "\n".join(l for l, _ in body_lines)
    md_text = re.sub(r"\n{4,}", "\n\n\n", md_text)
    (out / f"{doc_id}.md").write_text(md_text, encoding="utf-8")
    (out / "sections.json").write_text(json.dumps(sections, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "raw_pages.json").write_text(json.dumps(raw_pages, ensure_ascii=False), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["archivo", "pagina", "seccion", "ancho_px", "alto_px", "bbox_pdf",
                                          "tabla_asociada", "ocr_texto"])
        w.writeheader()
        w.writerows(images)
    if not images:
        img_dir.rmdir()

    n_tables = len(re.findall(r"^\|\s*:?-{3,}", md_text, re.MULTILINE))
    by_state = lambda s: [x["num"] for x in sections if x["estado"] == s]  # noqa: E731
    report = {
        "doc_id": doc_id, "archivo": meta["archivo"], "fabricante": fabricante, "producto": meta["producto"],
        "version": meta.get("version", ""), "paginas": len(doc), "paginas_escaneadas": scanned,
        "tablas_md": n_tables, "imagenes": len(images), "imagenes_con_ocr": sum(1 for i in images if i["ocr_texto"]),
        "secciones_presentes": len(by_state("presente")),
        "ausentes_original_probable": by_state("ausente_en_original_probable"),
        "fallo_extraccion": by_state("fallo_extraccion"), "no_determinado": by_state("no_determinado"),
        "segundos": round(time.time() - t0, 1),
    }
    return meta, report


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["archivo"]: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--only", help="procesar solo archivos cuya ruta contenga este texto")
    ap.add_argument("--solo-secciones", action="store_true",
                    help="recalcular secciones y reporte sin volver a convertir los PDF")
    ap.add_argument("--resumen", action="store_true", help="imprimir el resumen global de la extracción")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.solo_secciones:
        n = refresh_metadata(cfg)
        print(f"Metadatos (fabricante, marca, producto, versión) actualizados en {n} documentos")
        rows, diag = recompute_sections(cfg)
        print(f"Secciones recalculadas en {len(rows)} documentos; {len(diag)} secciones no presentes "
              f"→ output/reports/secciones_no_presentes.csv")
        summarize(cfg)
        return
    if args.resumen:
        summarize(cfg)
        return
    pdf_dir = Path(cfg["paths"]["pdf_dir"])
    pdfs = sorted(p for p in pdf_dir.rglob("*") if p.suffix.lower() == ".pdf")
    if args.only:
        pdfs = [p for p in pdfs if args.only.lower() in str(p).lower()]
    if not pdfs:
        raise SystemExit(f"No hay PDF en {pdf_dir}. Organice: data/pdfs/<carpeta>/<archivo>.pdf")

    man_path = Path(cfg["paths"]["manifest"])
    manifest = load_manifest(man_path)
    reports = []
    for pdf in pdfs:
        rel = str(pdf.relative_to(pdf_dir))
        row = manifest.get(rel, {"archivo": rel})
        row["archivo"] = rel
        if not row.get("doc_id"):
            base = f"{slug(pdf.parent.name)}__{slug(pdf.stem)}"
            used = {r.get("doc_id") for r in manifest.values()} | {r.get("doc_id") for r in reports}
            cand, n = base, 2
            while cand in used:
                cand, n = f"{base}_{n}", n + 1
            row["doc_id"] = cand
        print(f"[extract] {rel}")
        try:
            meta, rep = process_pdf(pdf, cfg, row)
        except Exception as e:  # noqa: BLE001 — se registra y se sigue con las demás
            print(f"  ERROR: {e}")
            reports.append({"doc_id": row.get("doc_id", ""), "archivo": rel, "error": repr(e)})
            continue
        manifest[rel] = {k: meta.get(k, "") for k in MANIFEST_COLS}
        reports.append(rep)
        print(f"  {rep['paginas']} págs, {rep['secciones_presentes']}/16 secciones, {rep['tablas_md']} tablas, "
              f"{rep['imagenes']} imágenes, ausentes={rep['ausentes_original_probable']} fallos={rep['fallo_extraccion']}")

    man_path.parent.mkdir(parents=True, exist_ok=True)
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(sorted(manifest.values(), key=lambda r: r["archivo"]))
    rep_dir = Path(cfg["paths"]["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in reports for k in r}, key=lambda k: list(reports[0]).index(k) if k in reports[0] else 99)
    with open(rep_dir / "extraction_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(reports)
    print(f"\nManifiesto: {man_path.relative_to(ROOT)}  |  Reporte: {(rep_dir / 'extraction_report.csv').relative_to(ROOT)}")
    print("Complete a mano en el manifiesto: procedencia, fecha_recepcion y corrija producto/versión si 'campos_auto' los marcó.")


if __name__ == "__main__":
    main()
