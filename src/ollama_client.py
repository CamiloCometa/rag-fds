"""Cliente mínimo de la API REST de Ollama (sin dependencias pesadas)."""
from __future__ import annotations

import time

import numpy as np
import requests


class Ollama:
    def __init__(self, base_url: str, timeout: int = 600):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def version(self) -> str:
        try:
            return requests.get(f"{self.base}/api/version", timeout=10).json().get("version", "?")
        except Exception:  # noqa: BLE001
            return "no disponible"

    def model_info(self, model: str) -> dict:
        try:
            r = requests.post(f"{self.base}/api/show", json={"model": model}, timeout=30).json()
            d = r.get("details", {})
            return {"model": model, "family": d.get("family"), "parameter_size": d.get("parameter_size"),
                    "quantization": d.get("quantization_level"), "digest": r.get("modified_at")}
        except Exception as e:  # noqa: BLE001
            return {"model": model, "error": str(e)}

    def embed(self, model: str, texts: list[str], batch_size: int = 16, options: dict | None = None,
              on_batch=None) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            body = {"model": model, "input": texts[i:i + batch_size], "truncate": True}
            if options:
                body["options"] = options
            r = requests.post(f"{self.base}/api/embed", json=body, timeout=self.timeout)
            r.raise_for_status()
            batch = r.json()["embeddings"]
            vecs += batch
            if on_batch:
                on_batch(i, batch)
        v = np.asarray(vecs, dtype=np.float32)
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)

    def chat(self, model: str, messages: list[dict], **options) -> dict:
        t0 = time.time()
        r = requests.post(f"{self.base}/api/chat", json={"model": model, "messages": messages, "stream": False,
                                                          "options": options}, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        return {"content": j["message"]["content"], "wall_s": round(time.time() - t0, 2),
                "prompt_tokens": j.get("prompt_eval_count"), "output_tokens": j.get("eval_count"),
                "load_s": round(j.get("load_duration", 0) / 1e9, 2),
                "eval_s": round(j.get("eval_duration", 0) / 1e9, 2)}
