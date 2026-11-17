"""Chunking por sección de la FDS con metadatos de trazabilidad.

Estrategia (documentar en el README/informe):
  * Unidad mínima = bloque (párrafo, tabla completa, imagen+OCR) con su página y sección.
  * Los bloques se agrupan en orden hasta `chunk_size` caracteres SIN cruzar el
    límite de una sección de la FDS (respect_sections).
  * Solapamiento: el fragmento siguiente arranca con los últimos bloques del
    anterior cuya longitud total sea <= `chunk_overlap` (o con la cola de texto
    del bloque si un único bloque supera el tamaño).
  * Tablas: se mantienen enteras; si una tabla supera chunk_size se corta por
    filas repitiendo la fila de encabezado en cada trozo.
  * Cada fragmento lleva una cabecera con fabricante, producto, versión y sección
    que se incluye en el texto embebido (mejora la discriminación entre productos).
"""
from __future__ import annotations

from pathlib import Path

from .common import norm
from .mdparse import blocks, read_md


def split_table(text: str, size: int):
    rows = text.splitlines()
    head = rows[:2] if len(rows) > 1 and set(rows[1].replace("|", "").strip()) <= set("-: ") else rows[:1]
    body = rows[len(head):]
    parts, cur = [], list(head)
    for r in body:
        if len("\n".join(cur + [r])) > size and len(cur) > len(head):
            parts.append("\n".join(cur))
            cur = list(head)
        cur.append(r)
    parts.append("\n".join(cur))
    return parts


def split_text(text: str, size: int, overlap: int):
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end), text.rfind(" ", start, end))
            end = cut + 1 if cut > start + size // 2 else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [o for o in out if o]


def header(meta: dict, sec: int, sec_name: str) -> str:
    s = f"Sección {sec}: {sec_name}" if sec else sec_name
    marca = f" (marca: {meta['marca']})" if meta.get("marca") and norm(meta["marca"]) not in norm(meta.get("fabricante", "")) else ""
    return (f"[Fabricante: {meta.get('fabricante', '')}{marca} | Producto: {meta.get('producto', '')} | "
            f"Versión: {meta.get('version', '') or 's/d'} | Carpeta: {meta.get('carpeta', '')} | {s}]")


def chunk_document(md_path: Path, size: int, overlap: int, respect_sections=True, keep_tables_whole=True):
    meta, body = read_md(md_path)
    units = []  # expandir bloques demasiado grandes
    for b in blocks(body):
        if len(b["text"]) <= size:
            units.append(b)
        elif b["kind"] == "table" and keep_tables_whole:
            units += [{**b, "text": t} for t in split_table(b["text"], size)]
        else:
            units += [{**b, "text": t} for t in split_text(b["text"], size, overlap)]

    chunks, cur = [], []

    def emit():
        if not cur or all(u["kind"] == "heading" for u in cur):
            return
        text = "\n\n".join(u["text"] for u in cur)
        sec, name = cur[-1]["section"], cur[-1]["section_name"]
        chunks.append({
            "doc_id": meta["doc_id"], "archivo": meta.get("archivo"), "fabricante": meta.get("fabricante"),
            "producto": meta.get("producto"), "version": meta.get("version"), "carpeta": meta.get("carpeta"), "marca": meta.get("marca"),
            "fecha_revision": meta.get("fecha_revision"),
            "pagina": cur[0]["page"], "pagina_fin": max(u.get("page_end", u["page"]) for u in cur),
            "seccion": sec, "seccion_nombre": name,
            "tiene_tabla": any(u["kind"] == "table" for u in cur),
            "ocr": any(u["page"] in (meta.get("paginas_escaneadas") or []) for u in cur),
            "imagenes": [l.split("](")[1].split(")")[0] for u in cur if u["kind"] == "image"
                         for l in u["text"].splitlines() if l.startswith("![")],
            "text": text,
            "header": header(meta, sec, name),
        })

    for u in units:
        new_section = cur and respect_sections and u["section"] != cur[-1]["section"]
        too_big = cur and len("\n\n".join(x["text"] for x in cur + [u])) > size
        if new_section or too_big:
            emit()
            if new_section:
                cur = []
            else:  # solapamiento con los últimos bloques del fragmento anterior
                tail, acc = [], 0
                for x in reversed(cur):
                    if acc + len(x["text"]) > overlap or x["kind"] == "heading":
                        break
                    tail.insert(0, x)
                    acc += len(x["text"])
                if not tail and overlap and cur[-1]["kind"] == "text":
                    t = cur[-1]["text"][-overlap:]
                    t = t[t.find(" ") + 1:] if " " in t else t
                    tail = [{**cur[-1], "text": t}]
                # conservar el encabezado de sección para dar contexto
                heads = [x for x in cur if x["kind"] == "heading"][:1]
                cur = heads + tail
        cur.append(u)
    emit()
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{c['doc_id']}#{i:04d}"
    return chunks
