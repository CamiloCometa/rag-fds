"""Demo web opcional (Gradio). Requiere: pip install -r requirements-demo.txt

    python app.py      → http://127.0.0.1:7860
"""
import gradio as gr

from src.rag import RAG

rag = RAG()
FABS = sorted({d["fabricante"] for d in rag.docs.values()})
def _label(d):
    return f"{d['fabricante']} · {d['producto']} (v{d['version'] or 's/d'}, {d.get('carpeta') or ''})"


PRODS = sorted(_label(d) for d in rag.docs.values())
LABEL2DOC = {_label(d): k for k, d in rag.docs.items()}


def ask(question, k, productos):
    docs = [LABEL2DOC[p] for p in productos] if productos else None
    r = rag.answer(question, int(k), docs)
    rows = [[s["n"], "sí" if s["citado"] else "", s["fabricante"], s["producto"], s["version"], s["documento"],
             s["pagina"] if s["pagina_fin"] in (None, s["pagina"]) else f"{s['pagina']}-{s['pagina_fin']}",
             f"{s['seccion']} {s['seccion_nombre']}", s["score"]] for s in r["sources"]]
    extra = ""
    if r.get("filter"):
        extra += f"\n\n_Búsqueda restringida a: {', '.join(r['filter'])}_"
    if r.get("warnings"):
        extra += "\n\n**Avisos:** " + "; ".join(r["warnings"])
    if r.get("usage"):
        extra += f"\n\n_{r['usage']['wall_s']} s · {r['usage']['prompt_tokens']} tokens de entrada_"
    return r["answer"] + extra, rows, "\n\n---\n\n".join(r["contexts"])


with gr.Blocks(title="RAG FDS pinturas") as demo:
    gr.Markdown(f"## Consulta de Fichas de Datos de Seguridad\n{len(rag.docs)} FDS indexadas · "
                f"{len(rag.chunks)} fragmentos · fabricantes: {', '.join(FABS)}")
    with gr.Row():
        q = gr.Textbox(label="Pregunta", scale=4)
        k = gr.Slider(1, 10, value=rag.cfg["retrieval"]["top_k"], step=1, label="top-k")
    prods = gr.Dropdown(PRODS, multiselect=True, label="Restringir a productos (opcional)")
    btn = gr.Button("Consultar", variant="primary")
    ans = gr.Markdown()
    src = gr.Dataframe(headers=["#", "citado", "fabricante", "producto", "versión", "documento", "página", "sección", "similitud"])
    with gr.Accordion("Fragmentos recuperados", open=False):
        ctx = gr.Markdown()
    btn.click(ask, [q, k, prods], [ans, src, ctx])
    q.submit(ask, [q, k, prods], [ans, src, ctx])

if __name__ == "__main__":
    demo.launch()
