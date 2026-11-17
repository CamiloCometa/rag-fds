"""Registra hardware, versiones y consumo (para el README/informe).

Uso: python -m src.hardware      → output/reports/hardware.json
Ejecútelo DESPUÉS de una corrida (con Ollama encendido y los modelos cargados) para
capturar la memoria que ocupan los modelos y los tiempos medios por consulta.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import requests

from .common import load_config, read_jsonl


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main():
    cfg = load_config()
    base = cfg["ollama"]["base_url"].rstrip("/")
    info = {"sistema": platform.platform(), "python": platform.python_version(), "cpu": platform.processor() or sh(["uname", "-m"]),
            "nucleos_logicos": os.cpu_count()}
    try:
        import psutil  # opcional
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        if Path("/proc/meminfo").exists():
            info["ram_gb"] = round(int(Path("/proc/meminfo").read_text().split()[1]) / 1e6, 1)
        elif shutil.which("sysctl"):
            info["ram_gb"] = round(int(sh(["sysctl", "-n", "hw.memsize"]) or 0) / 1e9, 1)
    try:  # EC2: tipo de instancia y zona vía IMDSv2 (no aplica fuera de AWS)
        r = requests.put("http://169.254.169.254/latest/api/token", timeout=1, proxies={"http": None},
                         headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        r.raise_for_status()
        tok = r.text

        def md(k):
            x = requests.get(f"http://169.254.169.254/latest/meta-data/{k}", timeout=1, proxies={"http": None},
                             headers={"X-aws-ec2-metadata-token": tok})
            x.raise_for_status()
            return x.text
        info["aws"] = {"instance_type": md("instance-type"), "zona": md("placement/availability-zone"),
                       "ami": md("ami-id")}
    except Exception:  # noqa: BLE001
        info["aws"] = None
    if shutil.which("nvidia-smi"):
        info["gpu"] = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    try:
        info["ollama_version"] = requests.get(f"{base}/api/version", timeout=5).json().get("version")
        tags = requests.get(f"{base}/api/tags", timeout=5).json().get("models", [])
        used = {cfg["generator"]["model"], cfg["embeddings"]["model"], cfg["evaluation"]["evaluator_model"]}
        info["modelos"] = [{"nombre": m["name"], "tamano_gb": round(m["size"] / 1e9, 2),
                            "cuantizacion": m.get("details", {}).get("quantization_level"),
                            "parametros": m.get("details", {}).get("parameter_size")}
                           for m in tags if m["name"] in used or m["name"].split(":")[0] in {u.split(":")[0] for u in used}]
        ps = requests.get(f"{base}/api/ps", timeout=5).json().get("models", [])
        info["modelos_cargados"] = [{"nombre": m["name"], "memoria_gb": round(m["size"] / 1e9, 2),
                                     "vram_gb": round(m.get("size_vram", 0) / 1e9, 2)} for m in ps]
    except Exception as e:  # noqa: BLE001
        info["ollama_error"] = str(e)

    ev_dir = Path(cfg["paths"]["eval_dir"])
    cons = {}
    for cname in cfg["evaluation"]["configs"]:
        p = ev_dir / cname / "responses.jsonl"
        if p.exists():
            rows = [r for r in read_jsonl(p) if r.get("usage")]
            if rows:
                n = len(rows)
                cons[cname] = {"consultas": n,
                               "seg_medio_por_respuesta": round(sum(r["usage"]["wall_s"] for r in rows) / n, 2),
                               "tokens_entrada_medio": round(sum(r["usage"]["prompt_tokens"] or 0 for r in rows) / n),
                               "tokens_salida_medio": round(sum(r["usage"]["output_tokens"] or 0 for r in rows) / n)}
        p = ev_dir / cname / "scores.jsonl"
        if p.exists():
            rows = read_jsonl(p)
            cons.setdefault(cname, {})["seg_evaluacion_ragas_total"] = round(
                sum(r.get(f"{m}_s", 0) for r in rows for m in ("faithfulness", "context_precision", "context_recall")), 1)
    idx = Path(cfg["paths"]["index_dir"]) / "index_meta.json"
    if idx.exists():
        cons["indexado"] = {k: json.loads(idx.read_text())[k] for k in ("n_fragmentos", "segundos_embeddings")}
    rep = Path(cfg["paths"]["reports_dir"]) / "extraction_report.csv"
    if rep.exists():
        import csv
        rows = list(csv.DictReader(open(rep, encoding="utf-8")))
        cons["extraccion"] = {"documentos": len(rows), "segundos_total": round(sum(float(r.get("segundos") or 0) for r in rows), 1)}
    info["consumo"] = cons
    out = Path(cfg["paths"]["reports_dir"]) / "hardware.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
