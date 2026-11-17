"""Evaluación con RAGAS (Faithfulness, Context Precision, Context Recall) con LLM evaluador local.

Subcomandos:
    python -m src.evaluate check      valida el conjunto de 12 preguntas (tipos, secciones, referencias, evidencia)
    python -m src.evaluate smoke      prueba el evaluador local con 2 casos antes de la corrida completa
    python -m src.evaluate generate   corre el RAG para las 12 preguntas con cada configuración (solo cambia top-k)
    python -m src.evaluate score      calcula las 3 métricas en las 10 preguntas respondibles de cada configuración
    python -m src.evaluate report     resumen por métrica, preguntas sin respuesta y casos para revisión manual
    python -m src.evaluate all        check + generate + score + report
    python -m src.evaluate dev --questions eval/dev_questions.yaml   prueba de desarrollo (sin puntuar)

Reglas que cumple el script:
  * Las respuestas de referencia NO se usan en el índice ni en el prompt del generador:
    solo se pasan al evaluador (Context Precision / Context Recall).
  * Corpus, chunking, embeddings, generador, prompt y evaluador son los mismos en ambas
    configuraciones; se verifica con el hash de configuración y el índice compartido.
  * Un fallo del evaluador se registra como NaN + mensaje de error; nunca como 0.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import platform
import sys
import time
from datetime import datetime
from importlib.metadata import version as pkg_version
from pathlib import Path

import yaml

from .common import ROOT, SECCIONES, config_hash, load_config, read_jsonl, write_jsonl

METRICS = ["faithfulness", "context_precision", "context_recall"]


# ------------------------------------------------------------------ preguntas
def load_questions(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_questions(qs: list[dict], strict=True) -> list[str]:
    errs = []
    tipos = [q.get("tipo") for q in qs]
    if len(qs) != 12:
        errs.append(f"Hay {len(qs)} preguntas; se requieren 12.")
    for t, n in (("directa", 6), ("integracion", 4), ("sin_respuesta", 2)):
        if tipos.count(t) != n:
            errs.append(f"Tipo '{t}': {tipos.count(t)} (se requieren {n}).")
    secs = {s for q in qs for s in (q.get("secciones") or [])}
    if len(secs) < 4:
        errs.append(f"Se cubren {len(secs)} secciones distintas; se requieren al menos 4.")
    ids = [q.get("id") for q in qs]
    if len(set(ids)) != len(ids):
        errs.append("IDs repetidos.")
    for q in qs:
        if not q.get("pregunta") or "TODO" in str(q.get("pregunta")):
            errs.append(f"{q.get('id')}: falta la pregunta.")
        if not q.get("productos") and not (q.get("tipo") == "sin_respuesta" and q.get("motivo_sin_respuesta")):
            errs.append(f"{q.get('id')}: identifique los productos consultados (campo productos).")
        if q.get("tipo") != "sin_respuesta":
            if not q.get("referencia") or "TODO" in str(q.get("referencia")):
                errs.append(f"{q.get('id')}: falta la respuesta de referencia verificada.")
            ev = q.get("evidencia") or []
            if not ev or any(not (e.get("doc_id") and e.get("pagina") and e.get("seccion") and e.get("fragmento")) for e in ev):
                errs.append(f"{q.get('id')}: evidencia incompleta (doc_id, pagina, seccion, fragmento).")
            if strict and not q.get("verificado_por"):
                errs.append(f"{q.get('id')}: falta verificado_por (quién la comparó con el PDF).")
    return errs


# ------------------------------------------------------------------ evaluador
def build_evaluator(cfg: dict):
    import instructor
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextPrecisionWithReference, ContextRecall, Faithfulness

    ev = cfg["evaluation"]
    client = AsyncOpenAI(api_key="ollama", base_url=cfg["ollama"]["base_url"].rstrip("/") + "/v1",
                         timeout=cfg["ollama"]["timeout"], max_retries=1)
    llm = llm_factory(ev["evaluator_model"], provider="openai", client=client,
                      mode=getattr(instructor.Mode, ev.get("evaluator_mode", "JSON")),
                      temperature=ev["evaluator_temperature"], max_tokens=ev["evaluator_max_tokens"])
    return {"faithfulness": Faithfulness(llm=llm),
            "context_precision": ContextPrecisionWithReference(llm=llm),
            "context_recall": ContextRecall(llm=llm)}


async def score_sample(metrics: dict, s: dict, retries: int) -> dict:
    out = {}
    for name, m in metrics.items():
        kwargs = {"user_input": s["user_input"], "retrieved_contexts": s["retrieved_contexts"]}
        if name == "faithfulness":
            kwargs["response"] = s["response"]
        else:
            kwargs["reference"] = s["reference"]
        err, t0 = None, time.time()
        for attempt in range(retries + 1):
            try:
                r = await m.ascore(**kwargs)
                val = float(r.value) if r.value is not None else math.nan
                if math.isnan(val):
                    raise ValueError("RAGAS devolvió NaN (salida del evaluador no interpretable)")
                out[name], err = val, None
                break
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:300]}"
        if err:
            out[name] = math.nan
        out[f"{name}_error"] = err or ""
        out[f"{name}_s"] = round(time.time() - t0, 1)
    return out


def evaluator_info(cfg: dict) -> dict:
    from .ollama_client import Ollama
    oll = Ollama(cfg["ollama"]["base_url"])
    pk = {}
    for p in ("ragas", "openai", "instructor", "pymupdf4llm", "pymupdf", "numpy"):
        try:
            pk[p] = pkg_version(p)
        except Exception:  # noqa: BLE001
            pk[p] = "?"
    ev = cfg["evaluation"]
    return {"fecha": datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0],
            "plataforma": platform.platform(), "paquetes": pk, "ollama_version": oll.version(),
            "evaluador": {**oll.model_info(ev["evaluator_model"]), "temperature": ev["evaluator_temperature"],
                          "max_tokens": ev["evaluator_max_tokens"], "modo_instructor": ev.get("evaluator_mode"),
                          "reintentos": ev.get("retries")},
            "generador": {**oll.model_info(cfg["generator"]["model"]), **cfg["generator"]},
            "embeddings": oll.model_info(cfg["embeddings"]["model"]),
            "config_hash": config_hash(cfg), "metricas": ["Faithfulness", "ContextPrecisionWithReference", "ContextRecall"]}


# ------------------------------------------------------------------ pasos
def _existing(path: Path, fresh: bool) -> dict:
    """Resultados ya guardados (para retomar si se cortó la sesión del Learner Lab)."""
    if fresh or not path.exists():
        return {}
    return {r["id"]: r for r in read_jsonl(path)}


def step_generate(cfg, qs, out_dir: Path, fresh=False, only=None, ids=None):
    from .rag import RAG
    rag = RAG()
    for cname, k in cfg["evaluation"]["configs"].items():
        if only and cname not in only:
            continue
        path = out_dir / cname / "responses.jsonl"
        done = _existing(path, fresh)
        rows = []
        for q in qs:
            if ids and q["id"] not in ids:  # trabajo repartido entre instancias: conservar lo que haya
                if q["id"] in done:
                    rows.append(done[q["id"]])
                continue
            prev = done.get(q["id"])
            if prev and prev["user_input"] == q["pregunta"] and prev.get("reference", "") == (q.get("referencia") or ""):
                rows.append(prev)
                print(f"[{cname}] {q['id']} ya generada (use --fresh para rehacer)")
                continue
            t0 = time.time()
            r = rag.answer(q["pregunta"], top_k=k)  # la referencia NO entra al generador
            rows.append({"id": q["id"], "tipo": q["tipo"], "config": cname, "top_k": k,
                         "user_input": q["pregunta"], "retrieved_contexts": r["contexts"], "response": r["answer"],
                         "reference": q.get("referencia") or "", "sources": r["sources"],
                         "insufficient_auto": r["insufficient"], "warnings": r.get("warnings", []),
                         "filter": r.get("filter"), "usage": r.get("usage"), "seconds": round(time.time() - t0, 1)})
            print(f"[{cname}] {q['id']} ({rows[-1]['seconds']} s) {r['answer'][:90]!r}")
            write_jsonl(path, rows + [d for i, d in done.items() if i not in {x["id"] for x in rows}
                                      and i in {qq["id"] for qq in qs}])  # guardado incremental
        write_jsonl(path, rows)
    (out_dir / "run_meta.json").write_text(json.dumps({"config_hash": config_hash(cfg), "configs": cfg["evaluation"]["configs"],
                                                       "index_meta": rag.meta}, ensure_ascii=False, indent=1), encoding="utf-8")


def step_score(cfg, out_dir: Path, only=None, limit=None, fresh=False, ids=None):
    metrics = build_evaluator(cfg)
    info = evaluator_info(cfg)
    (out_dir / "evaluator_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    for cname in (only or cfg["evaluation"]["configs"]):
        rows = read_jsonl(out_dir / cname / "responses.jsonl")
        path = out_dir / cname / ("scores_smoke.jsonl" if limit else "scores.jsonl")
        done = _existing(path, fresh or bool(limit))
        scored = []
        answerable = [r for r in rows if r["tipo"] != "sin_respuesta"][: limit or None]
        for r in answerable:
            prev = done.get(r["id"])
            if ids and r["id"] not in ids:
                if prev:
                    scored.append(prev)
                continue
            if prev and prev["response"] == r["response"] and prev["reference"] == r["reference"]:
                scored.append(prev)  # ya puntuada con la misma respuesta: no se repite
                print(f"[{cname}] {r['id']} ya puntuada (use --fresh para rehacer)")
                continue
            t0 = time.time()
            sc = asyncio.run(score_sample(metrics, r, cfg["evaluation"].get("retries", 1)))
            scored.append({**r, **sc})
            write_jsonl(path, scored)  # guardado incremental: si se corta la sesión, se retoma desde aquí
            print(f"[{cname}] {r['id']} ({time.time() - t0:.0f} s) " +
                  " ".join(f"{m}={sc[m]:.3f}" if not math.isnan(sc[m]) else f"{m}=FALLO" for m in METRICS))
        write_jsonl(path, scored)


def step_merge(cfg, out_dir: Path, other: Path):
    """Une resultados de otra instancia (otra carpeta output/eval) con los locales, por id y configuración.
    Prefiere la fila con más métricas válidas."""
    def nvalid(r):
        return sum(1 for m in METRICS if isinstance(r.get(m), (int, float)) and not math.isnan(r[m]))
    for cname in cfg["evaluation"]["configs"]:
        for fname in ("responses.jsonl", "scores.jsonl"):
            rows = {}
            for base in (out_dir, other):
                f = base / cname / fname
                if f.exists():
                    for r in read_jsonl(f):
                        cur = rows.get(r["id"])
                        if cur is None or nvalid(r) > nvalid(cur):
                            rows[r["id"]] = r
            if rows:
                write_jsonl(out_dir / cname / fname, sorted(rows.values(), key=lambda r: r["id"]))
                print(f"[merge] {cname}/{fname}: {len(rows)} filas")


def _fmt(v):
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else (round(v, 3) if isinstance(v, float) else v)


def step_report(cfg, qs, out_dir: Path):
    qmap = {q["id"]: q for q in qs}
    all_rows, summary = [], []
    for cname in cfg["evaluation"]["configs"]:
        p = out_dir / cname / "scores.jsonl"
        if not p.exists():
            print(f"Falta {p}; ejecute score.")
            continue
        rows = read_jsonl(p)
        all_rows += rows
        for m in METRICS:
            vals = [r[m] for r in rows if r[m] is not None and not math.isnan(r[m])]
            summary.append({"config": cname, "top_k": rows[0]["top_k"], "metrica": m,
                            "promedio": round(sum(vals) / len(vals), 3) if vals else "",
                            "casos_validos": len(vals), "fallos_evaluacion": len(rows) - len(vals), "n": len(rows)})
    # resultados por pregunta (CSV plano + JSON completo)
    flat = []
    for r in all_rows:
        flat.append({"id": r["id"], "config": r["config"], "top_k": r["top_k"], "tipo": r["tipo"],
                     "user_input": r["user_input"], "response": r["response"], "reference": r["reference"],
                     "retrieved_contexts": "\n<<<FRAGMENTO>>>\n".join(r["retrieved_contexts"]),
                     "fuentes": "; ".join(f"[{s['n']}] {s['producto']} p.{s['pagina']} s.{s['seccion']}"
                                          + ("*" if s["citado"] else "") for s in r["sources"]),
                     **{m: _fmt(r[m]) for m in METRICS}, **{f"{m}_error": r.get(f"{m}_error", "") for m in METRICS}})
    _write_csv(out_dir / "results.csv", flat)
    (out_dir / "results.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    _write_csv(out_dir / "summary.csv", summary)

    # preguntas sin respuesta (revisión aparte)
    noans = []
    for cname in cfg["evaluation"]["configs"]:
        for r in read_jsonl(out_dir / cname / "responses.jsonl"):
            if r["tipo"] == "sin_respuesta":
                noans.append({"id": r["id"], "config": cname, "pregunta": r["user_input"], "respuesta": r["response"],
                              "reconoce_falta_evidencia_auto": r["insufficient_auto"],
                              "inventa_informacion_manual": "", "comentario": ""})
    _write_csv(out_dir / "sin_respuesta.csv", noans)

    # casos para revisión manual: los 2 de menor puntaje medio + 1 adicional
    def mean_score(r):
        v = [r[m] for m in METRICS if r[m] is not None and not math.isnan(r[m])]
        return sum(v) / len(v) if v else -1
    ranked = sorted(all_rows, key=mean_score)
    chosen, seen = [], set()
    for r in ranked:
        if r["id"] not in seen:
            chosen.append(r)
            seen.add(r["id"])
        if len(chosen) == 2:
            break
    rest = [r for r in sorted(all_rows, key=mean_score, reverse=True) if r["id"] not in seen]
    if rest:
        chosen.append(rest[len(rest) // 2])
    cases = []
    for r in chosen:
        for s in r["sources"]:
            if s["citado"]:
                cases.append({"id": r["id"], "config": r["config"], "puntaje_medio": round(mean_score(r), 3),
                              "cita": s["n"], "fabricante": s["fabricante"], "producto": s["producto"],
                              "documento": s["documento"], "pagina": s["pagina"], "seccion": s["seccion"],
                              "evidencia_referencia": json.dumps(qmap.get(r["id"], {}).get("evidencia", []), ensure_ascii=False),
                              "respuesta": r["response"], "cita_coincide_con_pdf_manual": "", "dato_exacto_manual": "",
                              "error_observado": ""})
    _write_csv(out_dir / "casos_revision.csv", cases)

    md = ["| Config | top-k | Métrica | Promedio | Casos válidos | Fallos |", "|---|---|---|---|---|---|"]
    md += [f"| {s['config']} | {s['top_k']} | {s['metrica']} | {s['promedio']} | {s['casos_validos']}/{s['n']} | {s['fallos_evaluacion']} |"
           for s in summary]
    md += ["", "| ID | Tipo | " + " | ".join(f"{m} ({c})" for c in cfg['evaluation']['configs'] for m in ('F', 'CP', 'CR')) + " |",
           "|---|---|" + "---|" * (3 * len(cfg['evaluation']['configs']))]
    by = {(r["id"], r["config"]): r for r in all_rows}
    for q in qs:
        if q["tipo"] == "sin_respuesta":
            continue
        cells = []
        for c in cfg["evaluation"]["configs"]:
            r = by.get((q["id"], c), {})
            cells += [str(_fmt(r.get(m)) if r.get(m) is not None and not (isinstance(r.get(m), float) and math.isnan(r.get(m))) else "fallo") for m in METRICS]
        md.append(f"| {q['id']} | {q['tipo']} | " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nArchivos en {out_dir.relative_to(ROOT)}: results.csv/json, summary.csv/md, sin_respuesta.csv, casos_revision.csv")


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "smoke", "generate", "score", "report", "all", "dev", "merge"])
    ap.add_argument("--configs", help="solo estas configuraciones, separadas por coma (p. ej. B_k6)")
    ap.add_argument("--ids", help="solo estas preguntas, separadas por coma (p. ej. Q01,Q02)")
    ap.add_argument("--desde", help="merge: carpeta output/eval traída de otra instancia")
    ap.add_argument("--questions")
    ap.add_argument("--no-strict", action="store_true", help="no exigir verificado_por")
    ap.add_argument("--fresh", action="store_true", help="rehacer todo en vez de retomar lo ya guardado")
    a = ap.parse_args()
    cfg = load_config()
    qpath = Path(a.questions) if a.questions else Path(ROOT / cfg["evaluation"]["questions"])
    qs = load_questions(qpath)
    out_dir = Path(cfg["paths"]["eval_dir"])

    only = a.configs.split(",") if a.configs else None
    ids = set(a.ids.split(",")) if a.ids else None
    if a.cmd == "dev":
        from .rag import RAG, print_answer
        rag = RAG()
        for q in qs:
            print(f"\n=== {q.get('id')}: {q['pregunta']}")
            print_answer(rag.answer(q["pregunta"]))
        return
    if a.cmd in ("check", "all"):
        errs = check_questions(qs, strict=not a.no_strict)
        secs = sorted({s for q in qs for s in (q.get("secciones") or [])})
        print(f"{len(qs)} preguntas; secciones cubiertas: {', '.join(f'{s} ({SECCIONES.get(s, '?')})' for s in secs)}")
        print("\n".join("  - " + e for e in errs) if errs else "Conjunto de preguntas válido.")
        if errs and a.cmd == "all":
            raise SystemExit("Corrija el conjunto de preguntas antes de evaluar.")
    if a.cmd == "smoke":
        first = next(iter(cfg["evaluation"]["configs"]))
        if not (out_dir / first / "responses.jsonl").exists():
            step_generate(cfg, [q for q in qs if q["tipo"] != "sin_respuesta"][:2], out_dir)
        step_score(cfg, out_dir, only=[first], limit=2)
    if a.cmd in ("generate", "all"):
        step_generate(cfg, qs, out_dir, fresh=a.fresh, only=only, ids=ids)
    if a.cmd in ("score", "all"):
        step_score(cfg, out_dir, only=only, fresh=a.fresh, ids=ids)
    if a.cmd == "merge":
        step_merge(cfg, out_dir, Path(a.desde))
    if a.cmd in ("report", "all"):
        step_report(cfg, qs, out_dir)


if __name__ == "__main__":
    main()
