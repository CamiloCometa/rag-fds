"""Construye el índice vectorial: fragmentos (JSONL) + embeddings normalizados (NumPy).

Elección: búsqueda exacta por producto punto sobre una matriz NumPy. Con decenas de
FDS (miles de fragmentos) es instantánea, no requiere servidor y es 100 %
reproducible; una base vectorial (Chroma, Qdrant) solo añadiría complejidad.

Uso: python -m src.index
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .chunking import chunk_document
from .common import ROOT, config_hash, load_config, slug, write_jsonl
from .ollama_client import Ollama


def main():
    cfg = load_config()
    ck, em = cfg["chunking"], cfg["embeddings"]
    md_root = Path(cfg["paths"]["md_dir"])
    out = Path(cfg["paths"]["index_dir"])
    out.mkdir(parents=True, exist_ok=True)

    chunks = []
    for d in sorted(p for p in md_root.iterdir() if p.is_dir()):
        md = d / f"{d.name}.md"
        if md.exists():
            cs = chunk_document(md, ck["chunk_size"], ck["chunk_overlap"], ck["respect_sections"], ck["keep_tables_whole"])
            chunks += cs
            print(f"[index] {d.name}: {len(cs)} fragmentos")
    if not chunks:
        raise SystemExit("No hay Markdown. Ejecute primero: python -m src.extract")

    oll = Ollama(cfg["ollama"]["base_url"], cfg["ollama"]["timeout"])
    texts = [c["header"] + "\n" + c["text"] for c in chunks]
    keys = [hashlib.sha1((em["model"] + "\x00" + t).encode()).hexdigest() for t in texts]
    # Caché persistente: si la sesión se corta, al relanzar solo se calcula lo que falta.
    cache_dir = out / "cache" / slug(em["model"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for f in sorted(cache_dir.glob("part_*.npz")):
        z = np.load(f)
        cache.update(zip(z["keys"].tolist(), z["vecs"]))
    todo = [i for i, k in enumerate(keys) if k not in cache]
    print(f"[index] {len(chunks)} fragmentos; {len(chunks) - len(todo)} ya en caché, faltan {len(todo)} "
          f"(modelo {em['model']})", flush=True)
    t0 = time.time()
    bs = em["batch_size"]
    opts = {"num_thread": em["num_thread"]} if em.get("num_thread") else None
    for start in range(0, len(todo), bs * 8):  # guardar cada 8 lotes
        idx = todo[start:start + bs * 8]
        v = oll.embed(em["model"], [texts[i] for i in idx], bs, opts)
        np.savez(cache_dir / f"part_{int(time.time() * 1000)}.npz", keys=np.array([keys[i] for i in idx]), vecs=v)
        cache.update(zip([keys[i] for i in idx], v))
        done = start + len(idx)
        el = time.time() - t0
        eta = el / done * (len(todo) - done)
        print(f"[index] {done}/{len(todo)} ({100 * done / len(todo):.0f} %) · {el / 60:.1f} min · "
              f"faltan ~{eta / 60:.0f} min", flush=True)
    emb = np.stack([cache[k] for k in keys]).astype(np.float32)
    secs = time.time() - t0
    np.save(out / "embeddings.npy", emb)
    write_jsonl(out / "chunks.jsonl", chunks)
    lens = [len(c["text"]) for c in chunks]
    meta = {
        "creado": datetime.now().isoformat(timespec="seconds"), "config_hash": config_hash(cfg),
        "chunking": ck, "embeddings": {**em, "dim": int(emb.shape[1]), "info": oll.model_info(em["model"])},
        "ollama_version": oll.version(), "n_fragmentos": len(chunks), "n_documentos": len({c["doc_id"] for c in chunks}),
        "longitud_caracteres": {"min": min(lens), "media": round(sum(lens) / len(lens)), "max": max(lens)},
        "segundos_embeddings": round(secs, 1),
    }
    (out / "index_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Índice: {len(chunks)} fragmentos de {meta['n_documentos']} FDS, dim={emb.shape[1]}, {secs:.1f}s "
          f"→ {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
