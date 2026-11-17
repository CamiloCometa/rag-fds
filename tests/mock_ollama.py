"""Servidor falso de Ollama SOLO para pruebas de integración del código (sin modelos).

Embeddings = hashing de trigramas; chat = devuelve la primera línea con datos del
fragmento [1]; /v1/chat/completions = instancia mínima válida del JSON schema que
pide instructor/RAGAS. Los puntajes que produce NO significan nada.

    python tests/mock_ollama.py 11434
"""
import hashlib
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

DIM = 256


def embed(t):
    v = np.zeros(DIM)
    t = re.sub(r"\s+", " ", t.lower())
    for i in range(len(t) - 2):
        v[int(hashlib.md5(t[i:i + 3].encode()).hexdigest(), 16) % DIM] += 1
    return (v / (np.linalg.norm(v) or 1)).tolist()


def instance(schema, defs):
    if "$ref" in schema:
        return instance(defs[schema["$ref"].split("/")[-1]], defs)
    if "anyOf" in schema:
        return instance(schema["anyOf"][0], defs)
    t = schema.get("type")
    if t == "object":
        return {k: instance(v, defs) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [instance(schema.get("items", {}), defs)]
    if t == "integer":
        return 1
    if t == "number":
        return 1.0
    if t == "boolean":
        return True
    if "enum" in schema:
        return schema["enum"][0]
    return "ok"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self.send({"version": "mock-0.0"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path == "/api/embed":
            inp = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return self.send({"embeddings": [embed(t) for t in inp]})
        if self.path == "/api/show":
            return self.send({"details": {"family": "mock", "parameter_size": "0B", "quantization_level": "none"}})
        if self.path == "/api/chat":
            user = body["messages"][-1]["content"]
            if "Pregunta: " in user and "no existe" in user.split("Pregunta: ")[-1].lower():
                ans = "No hay evidencia suficiente en las FDS recuperadas para responder."
            else:
                frag = user.split("[1] ", 1)[1].split("\n\n[2]")[0].splitlines()
                line = next((l for l in frag[1:] if re.search(r"\d", l) and not l.startswith("#")), frag[-1])
                ans = f"Según la FDS: {line.strip()} [1]"
            return self.send({"message": {"content": ans}, "prompt_eval_count": len(user) // 4, "eval_count": 30,
                              "load_duration": 0, "eval_duration": 1e8})
        if self.path.startswith("/v1/chat/completions"):
            txt = " ".join(m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
                           for m in body["messages"])
            m = re.search(r"json_schema:\s*(\{.*\})\s*Make sure", txt, re.S) or re.search(r"(\{\"\$defs\".*\}|\{\"properties\".*\})", txt, re.S)
            schema = json.loads(m.group(1)) if m else {}
            content = json.dumps(instance(schema, schema.get("$defs", {})))
            return self.send({"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model"),
                              "choices": [{"index": 0, "finish_reason": "stop",
                                           "message": {"role": "assistant", "content": content}}],
                              "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        self.send_response(404)
        self.end_headers()


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 11434), H).serve_forever()
