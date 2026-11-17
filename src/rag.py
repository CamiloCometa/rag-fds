"""Recuperación + generación con citas trazables.

    from src.rag import RAG
    rag = RAG()                       # carga config.yaml e índice
    out = rag.answer("¿Cuál es el punto de inflamación del Esmalte X?", top_k=3)

CLI:
    python -m src.rag "pregunta" [--k 4] [--fabricante ColorSur] [--producto "Vinilo"] [--json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .common import SECCIONES, load_config, norm, read_jsonl
from .ollama_client import Ollama

INSUFFICIENT = "No hay evidencia suficiente en las FDS recuperadas para responder"

SYSTEM_PROMPT = f"""Eres un asistente técnico que responde preguntas sobre Fichas de Datos de Seguridad (FDS) de pinturas.
Usa EXCLUSIVAMENTE los fragmentos numerados que se te entregan. Reglas obligatorias:
1. Cada oración con información debe terminar con la(s) cita(s) del fragmento que la respalda, en el formato [n] o [n][m].
2. Copia valores, unidades, números CAS y frases H/P exactamente como aparecen. No conviertas unidades ni redondees.
3. No uses conocimiento externo ni completes información por inferencia. Si un dato no está en los fragmentos, dilo.
4. Indica siempre a qué fabricante y producto corresponde cada dato. Si la pregunta compara productos, responde por producto.
5. Si los fragmentos no permiten responder la pregunta, responde exactamente: "{INSUFFICIENT}." y en una frase indica qué información faltaría.
6. Responde en español, de forma breve y precisa.

Escribe oraciones completas y naturales (sin plantillas ni marcadores) que nombren el producto y el fabricante tal como aparecen en los fragmentos, y termina cada oración con la cita del fragmento, por ejemplo [2]."""

REMINDER = "Responde en oraciones completas usando solo los fragmentos; termina cada oración con su cita entre corchetes."


_TOK = re.compile(r"\b\d{2,7}-\d{2}-\d\b|\b[HP]\d{3}\b|\bUN\s?\d{4}\b|\d+(?:[.,]\d+)?|[a-záéíóúñ]{5,}", re.IGNORECASE)


def attach_citations(ans: str, ctx: list[dict]):
    """Si el modelo no citó, añade a cada oración la cita del fragmento con mayor coincidencia de datos
    (números, CAS, códigos H/P/UN y palabras largas). Solo cita si hay al menos un dato numérico/código
    compartido; si no, la oración queda sin cita (y se reporta)."""
    sents = re.split(r"(?<=[.!?])\s+", ans.strip())
    texts = [norm(c["text"]) for c in ctx]
    out, used = [], set()
    for snt in sents:
        toks = {norm(t) for t in _TOK.findall(snt)}
        hard = {t for t in toks if any(ch.isdigit() for ch in t)}
        best, best_sc = None, 0
        for i, tx in enumerate(texts, start=1):
            sc = 3 * sum(t in tx for t in hard) + sum(t in tx for t in toks - hard)
            if sc > best_sc and any(t in tx for t in hard):
                best, best_sc = i, sc
        if best:
            snt = re.sub(r"([.!?]?)$", lambda m: f" [{best}]" + m.group(1), snt, count=1)
            used.add(best)
        out.append(snt)
    return " ".join(out), sorted(used)


def format_context(i: int, c: dict) -> str:
    pag = f"{c['pagina']}" + (f"-{c['pagina_fin']}" if c.get("pagina_fin") and c["pagina_fin"] != c["pagina"] else "")
    return (f"[{i}] Fabricante: {c['fabricante']}" + (f" (marca: {c['marca']})" if c.get("marca") else "") + f" | Producto: {c['producto']} | Versión: {c.get('version') or 's/d'} | "
            f"Carpeta: {c.get('carpeta') or ''} | Documento: {c['archivo']} | Página: {pag} | Sección {c['seccion']}: {c['seccion_nombre']}"
            + (" | Texto obtenido por OCR" if c.get("ocr") else "") + f"\n{c['text']}")


SECTION_HINTS = {
    2: ["frases h", "indicaciones de peligro", "indicacion de peligro", "pictograma", "palabra de advertencia",
        "clasificacion", "peligros"],
    3: ["cas", "composicion", "componente", "concentracion", "contiene", "ingrediente"],
    4: ["primeros auxilios"],
    5: ["extincion", "extintor", "incendio"],
    6: ["derrame", "vertido", "fuga"],
    7: ["almacenamiento", "almacenar", "manipulacion"],
    8: ["proteccion", "respirador", "respiratoria", "guantes", "epp", "limite de exposicion", "tlv", "twa",
        "exposicion", "c.o.v", "cov", "compuestos organicos volatiles"],
    9: ["punto de inflamacion", "inflamacion", "densidad", " ph", "viscosidad", "ebullicion", "solubilidad",
        "c.o.v", "cov", "compuestos organicos volatiles", "estado fisico", "olor"],
    10: ["estabilidad", "reactividad", "incompatib"],
    11: ["toxic", "ld50", "dl50", "cl50", "lc50"],
    12: ["ecolog", "peces", "acuatic", "medio ambiente"],
    13: ["eliminacion", "disposicion final", "residuos"],
    14: ["onu", " un ", "transporte", "grupo de embalaje", "clase de peligro"],
    15: ["reglament", "normativ", "legislacion"],
}


def section_hints(question: str) -> set[int]:
    q = " " + norm(question) + " "
    return {sec for sec, ws in SECTION_HINTS.items() if any(w in q for w in ws)}


_STOP = set("""de la el los las y o u en del al a un una unos unas por para con sin sobre que se es son su sus
lo le les como cual cuál cuales qué cuál cómo entre según este esta estos estas ese esa fds ficha datos seguridad""".split())


class BM25:
    """BM25 léxico (k1=1.5, b=0.75) sobre los mismos fragmentos: captura coincidencias exactas
    (nombres de producto, N.º CAS, códigos H/P, UN) que un modelo de embeddings pequeño puede perder."""

    def __init__(self, docs, k1=1.5, b=0.75):
        from collections import Counter
        self.k1, self.b = k1, b
        self.tf = [Counter(self.tok(d)) for d in docs]
        self.len = np.array([sum(t.values()) for t in self.tf], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(self.len) else 1.0
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: float(np.log(1 + (n - c + 0.5) / (c + 0.5))) for w, c in df.items()}
        self.post = {}
        for i, t in enumerate(self.tf):
            for w in t:
                self.post.setdefault(w, []).append(i)

    @staticmethod
    def tok(text):
        return [w for w in re.findall(r"[a-z0-9]+(?:[-+][a-z0-9]+)*", norm(text)) if w not in _STOP and len(w) > 1]

    def scores(self, query):
        s = np.zeros(len(self.tf), dtype=np.float32)
        for w in set(self.tok(query)):
            idf = self.idf.get(w)
            if not idf:
                continue
            for i in self.post[w]:
                f = self.tf[i][w]
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class RAG:
    def __init__(self, config_path=None):
        self.cfg = load_config(config_path)
        idx = Path(self.cfg["paths"]["index_dir"])
        if not (idx / "embeddings.npy").exists():
            raise SystemExit("No hay índice. Ejecute: python -m src.index")
        self.chunks = read_jsonl(idx / "chunks.jsonl")
        self.emb = np.load(idx / "embeddings.npy")
        self.meta = json.loads((idx / "index_meta.json").read_text(encoding="utf-8"))
        self.oll = Ollama(self.cfg["ollama"]["base_url"], self.cfg["ollama"]["timeout"])
        if self.meta["embeddings"]["model"] != self.cfg["embeddings"]["model"]:
            raise SystemExit("El modelo de embeddings del índice no coincide con config.yaml; reconstruya el índice.")
        name2num = {v: k for k, v in SECCIONES.items()}
        self.canon = np.array([name2num.get(c.get("seccion_nombre"), 0) for c in self.chunks])
        self.bm25 = BM25([c["header"] + "\n" + c["text"] for c in self.chunks])
        self.docs = {}
        for c in self.chunks:
            self.docs.setdefault(c["doc_id"], {k: c.get(k) for k in ("doc_id", "fabricante", "marca", "producto", "version", "archivo", "carpeta")})

    # ---------- detección de productos / fabricantes mencionados
    def detect_docs(self, question: str) -> list[str]:
        q = " " + norm(question) + " "
        prods, fabs = set(), set()
        for d in self.docs.values():
            p = norm(d["producto"])
            toks = [t for t in re.findall(r"\w+", p) if len(t) > 2
                    and t not in {"para", "con", "tipo", "blanco", "pintura", "col", "version", "parte", "part", "hoja", "seguridad"}
                    and not (t.isdigit() and len(t) >= 6)]  # códigos de referencia del nombre de archivo
            if p and (f" {p} " in q or (len(toks) >= 2 and sum(f" {t} " in q for t in toks) >= max(2, int(0.75 * len(toks))))):
                prods.add(d["doc_id"])
            for f in (norm(d["fabricante"]), norm(d.get("marca") or "")):
                f = re.sub(r"\b(s\.?a\.?s?|ltda|s\.?a)\.?$", "", f).strip(" .")
                words = [w for w in f.split() if len(w) > 2 and w not in {"colombia", "de", "pinturas", "compania", "global"}]
                if f and (f" {f} " in q or any(f" {w} " in q for w in words)):
                    fabs.add(d["doc_id"])
        self._match_kind = "producto" if prods else ("fabricante" if fabs else "")
        return sorted(prods) or sorted(fabs)

    # ---------- recuperación
    def retrieve(self, question: str, top_k: int, doc_ids: list[str] | None = None):
        em = self.cfg["embeddings"]
        opts = {"num_thread": em["num_thread"]} if em.get("num_thread") else None
        qv = self.oll.embed(em["model"], [question], options=opts)[0]
        dense = self.emb @ qv
        mode = self.cfg["retrieval"].get("mode", "hybrid")
        if mode == "dense":
            scores = dense
        else:
            bm = self.bm25.scores(question)
            if mode == "bm25":
                scores = bm
            else:  # híbrido: fusión por rangos recíprocos (RRF, k=60) de denso + BM25 (+ prior de sección)
                rr = np.zeros(len(dense))
                lists = [dense, bm]
                secs = section_hints(question)
                if secs and self.cfg["retrieval"].get("section_prior", True):
                    # la FDS tiene 16 secciones estándar: si la pregunta pide un dato típico de una sección,
                    # los fragmentos de esa sección suben (lista adicional en la fusión RRF)
                    lists.append(dense + 10.0 * np.isin(self.canon, list(secs)))
                for arr in lists:
                    order_ = np.argsort(-arr)
                    ranks = np.empty(len(arr))
                    ranks[order_] = np.arange(1, len(arr) + 1)
                    rr += 1.0 / (60 + ranks)
                scores = rr
        self._last_dense = dense
        kind = "manual"
        if doc_ids is None and self.cfg["retrieval"].get("auto_filter"):
            doc_ids = self.detect_docs(question) or None
            kind = self._match_kind
        allowed = set(doc_ids or [])
        products = {norm(self.docs[d]["producto"]) for d in allowed if d in self.docs}
        if allowed and (kind == "fabricante" or len(allowed) == 1):
            # solo se nombró el fabricante, o una sola FDS: top-k por similitud dentro de esas FDS
            order = [i for i in np.argsort(-scores) if self.chunks[i]["doc_id"] in allowed][:top_k]
        elif allowed:
            # comparación: reparte k entre productos (o entre versiones/años si es el mismo producto),
            # empezando por el grupo con mejor similitud
            key = (lambda i: norm(self.chunks[i]["producto"])) if len(products) > 1 else (lambda i: self.chunks[i]["doc_id"])
            ranked = [i for i in np.argsort(-scores) if self.chunks[i]["doc_id"] in allowed]
            by_prod = {}
            for i in ranked:
                by_prod.setdefault(key(i), []).append(i)
            groups = dict(sorted(by_prod.items(), key=lambda kv: -scores[kv[1][0]]))
            order, r = [], 0
            while len(order) < top_k and any(r < len(g) for g in groups.values()):
                for g in groups.values():
                    if r < len(g) and len(order) < top_k:
                        order.append(g[r])
                r += 1
            order = sorted(order, key=lambda i: -scores[i])
        if not allowed:
            order = list(np.argsort(-scores)[:top_k])
        res = [{**self.chunks[i], "score": round(float(dense[i]), 4), "rank": n + 1} for n, i in enumerate(order)]
        thr = self.cfg["retrieval"].get("min_score")
        if thr is not None:
            res = [r for r in res if r["score"] >= thr]
        return res, doc_ids or []

    # ---------- generación
    def answer(self, question: str, top_k: int | None = None, doc_ids=None) -> dict:
        k = top_k or self.cfg["retrieval"]["top_k"]
        ctx, filt = self.retrieve(question, k, doc_ids)
        g = self.cfg["generator"]
        if not ctx:
            return {"question": question, "answer": INSUFFICIENT + " (ningún fragmento supera el umbral).",
                    "insufficient": True, "contexts": [], "sources": [], "filter": filt, "top_k": k}
        context_txt = "\n\n".join(format_context(i, c) for i, c in enumerate(ctx, start=1))
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Fragmentos recuperados:\n\n{context_txt}\n\nPregunta: {question}\n\n{REMINDER}"}]
        extra = {"num_thread": g["num_thread"]} if g.get("num_thread") else {}
        out = self.oll.chat(g["model"], msgs, temperature=g["temperature"], seed=g["seed"],
                            num_ctx=g["num_ctx"], num_predict=g["num_predict"], **extra)
        ans = re.sub(r"^\s*(\[\d+\]\s*)+", "", out["content"].strip())  # quitar cita suelta al inicio
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", ans)})
        valid = [n for n in cited if 1 <= n <= len(ctx)]
        auto_cited = False
        if not valid and norm(INSUFFICIENT) not in norm(ans):
            ans, valid = attach_citations(ans, ctx)
            auto_cited = bool(valid)
        sources = [{"n": i, "doc_id": c["doc_id"], "fabricante": c["fabricante"], "producto": c["producto"],
                    "version": c.get("version"), "carpeta": c.get("carpeta"), "documento": c["archivo"], "pagina": c["pagina"],
                    "pagina_fin": c.get("pagina_fin"), "seccion": c["seccion"], "seccion_nombre": c["seccion_nombre"],
                    "score": c["score"], "citado": i in valid, "chunk_id": c["chunk_id"]}
                   for i, c in enumerate(ctx, start=1)]
        insufficient = norm(INSUFFICIENT) in norm(ans)
        warnings = []
        if set(cited) - set(valid):
            warnings.append(f"citas inexistentes: {sorted(set(cited) - set(valid))}")
        if auto_cited:
            warnings.append("el modelo no citó: citas [n] añadidas automáticamente al fragmento que contiene los datos de la oración")
        if not valid and not insufficient:
            warnings.append("respuesta sin citas")
        return {"question": question, "answer": ans, "insufficient": insufficient, "top_k": k, "filter": filt,
                "contexts": [format_context(i, c) for i, c in enumerate(ctx, start=1)],
                "sources": sources, "warnings": warnings, "usage": out}


def print_answer(r: dict):
    print("\n" + r["answer"] + "\n")
    if r.get("filter"):
        print(f"(búsqueda restringida a: {', '.join(r['filter'])})")
    print("Fuentes:")
    for s in r["sources"]:
        mark = "*" if s["citado"] else " "
        pag = s["pagina"] if s["pagina_fin"] in (None, s["pagina"]) else f"{s['pagina']}-{s['pagina_fin']}"
        print(f" {mark}[{s['n']}] {s['fabricante']} · {s['producto']} (v{s['version'] or 's/d'}) · {s['documento']} · "
              f"pág. {pag} · Sección {s['seccion']} {s['seccion_nombre']} · sim={s['score']}")
    for w in r.get("warnings", []):
        print(f"AVISO: {w}")
    if r.get("usage"):
        u = r["usage"]
        print(f"[{u['wall_s']} s, {u['prompt_tokens']} tokens entrada, {u['output_tokens']} salida]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--k", type=int)
    ap.add_argument("--fabricante")
    ap.add_argument("--producto")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rag = RAG()
    docs = None
    if a.fabricante or a.producto:
        docs = [d for d, m in rag.docs.items()
                if (not a.fabricante or norm(a.fabricante) in norm(m["fabricante"]))
                and (not a.producto or norm(a.producto) in norm(m["producto"]))]
        if not docs:
            raise SystemExit("Ninguna FDS coincide con el filtro.")
    qs = [a.question] if a.question else iter(lambda: input("\nPregunta (vacío para salir): ").strip(), "")
    for q in qs:
        r = rag.answer(q, a.k, docs)
        print(json.dumps(r, ensure_ascii=False, indent=1) if a.json else "", end="")
        if not a.json:
            print_answer(r)


if __name__ == "__main__":
    main()
