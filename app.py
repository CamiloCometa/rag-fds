"""Demo web del RAG sobre FDS (Gradio).

Requiere:  pip install -r requirements-demo.txt

    python app.py            -> http://<IP-del-servidor>:7860  (escucha en 0.0.0.0)
    python app.py --share    -> además crea un enlace público temporal https://xxxx.gradio.live
"""
import html
import re
import sys

import gradio as gr

from src.rag import RAG

EXAMPLES = [
    "¿Cuál es el punto de inflamación del ESMALTE TOP QUALITY NEGRO 1772 de Pintuco?",
    "¿Cuál es el número ONU para el transporte del PINTUTRAFICO ALQUIDICO AMARILLO 659?",
    "¿Qué medios de extinción recomienda la FDS del ESMALTE PROTEKLAND BLANCO 2 en 1 de Pintuland y cuál no se recomienda?",
    "Compara el punto de inflamación de la PINTULACA BLANCO 7519 y del ESMALTE PINTULUX ALUMINIO de Pintuco. ¿Cuál es más inflamable?",
    "¿Cuál es el rendimiento en metros cuadrados por galón del VINILTEX PRO 650 BLANCO de Pintuco?",
]

CSS = """
:root{--fds-red:#c4182b;--fds-ink:#16202b;--fds-bar:#1f2d3b;--fds-line:#d5d8d4;--fds-muted:#5a6470;--fds-ok:#1e7a46}
.gradio-container{max-width:1180px !important;margin:0 auto !important}
#hdr{border-bottom:6px solid var(--fds-red);padding:6px 4px 16px;margin-bottom:6px}
#hdr .eb{font:500 12px/1.4 "IBM Plex Mono",monospace;letter-spacing:.06em;text-transform:uppercase;color:var(--fds-muted)}
#hdr h1{font:700 34px/1.15 "IBM Plex Sans Condensed","Arial Narrow",sans-serif;margin:6px 0 8px;color:var(--body-text-color)}
#hdr p{margin:0;color:var(--body-text-color-subdued);max-width:75ch}
#hdr .stats{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
#hdr .stats span{border:1px solid var(--fds-line);padding:3px 10px;font-size:13px;border-radius:2px}
#hdr .stats b{font-family:"IBM Plex Sans Condensed",sans-serif}
.secbar{background:var(--fds-bar);color:#fff;padding:7px 12px;font:600 16px "IBM Plex Sans Condensed",sans-serif;margin:4px 0 8px}
.secbar small{font:500 11px "IBM Plex Mono",monospace;opacity:.8;margin-right:10px;letter-spacing:.06em}
.ans{border:1px solid var(--fds-line);border-left:6px solid var(--fds-ok);padding:14px 16px;background:var(--block-background-fill);font-size:17px;line-height:1.55}
.ans.no{border-left-color:var(--fds-red)}
.ans .pill{display:inline-block;font:500 12px "IBM Plex Mono",monospace;padding:2px 9px;border:1px solid currentColor;border-radius:20px;margin-bottom:8px;color:var(--fds-ok)}
.ans.no .pill{color:var(--fds-red)}
.ans sup.c{background:var(--fds-red);color:#fff;font:600 11px "IBM Plex Mono",monospace;padding:1px 5px;border-radius:3px;margin-left:2px}
.meta{font-size:13px;color:var(--body-text-color-subdued);margin-top:8px}
.warn{font-size:13px;color:#a35b00;margin-top:6px}
.src{display:grid;gap:8px;margin-top:4px}
.card{border:1px solid var(--fds-line);padding:10px 12px;display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:start;background:var(--block-background-fill)}
.card .n{font:600 14px "IBM Plex Mono",monospace;background:var(--fds-bar);color:#fff;padding:3px 8px;border-radius:2px}
.card.cit .n{background:var(--fds-red)}
.card .t{font-weight:600}
.card .d{font-size:13px;color:var(--body-text-color-subdued)}
.card .s{font:500 12px "IBM Plex Mono",monospace;color:var(--body-text-color-subdued);text-align:right;white-space:nowrap}
.card .tag{display:inline-block;font:500 11px "IBM Plex Mono",monospace;color:var(--fds-red);border:1px solid var(--fds-red);padding:0 6px;border-radius:10px;margin-top:4px}
.empty{color:var(--body-text-color-subdued);font-size:15px;padding:18px;border:1px dashed var(--fds-line)}
#foot{font-size:13px;color:var(--body-text-color-subdued);border-top:1px solid var(--fds-line);padding-top:10px;margin-top:10px}
"""

HEAD = ('<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@600;700'
        '&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">')

rag = RAG()


def _label(d):
    return f"{d['fabricante']} · {d['producto']} (v{d['version'] or 's/d'}, {d.get('carpeta') or ''})"


PRODS = sorted(_label(d) for d in rag.docs.values())
LABEL2DOC = {_label(d): k for k, d in rag.docs.items()}
FABS = sorted({d["fabricante"] for d in rag.docs.values()})

HEADER = f"""
<div id="hdr">
  <div class="eb">Parcial final · NLP · Sofia Londoño, Mateo Arrubla, Leonardo Diazgranados e Ivan Camilo Cometa</div>
  <h1>Consulta de Fichas de Datos de Seguridad</h1>
  <p>Pregunta por cualquier dato de las FDS de pinturas. Cada respuesta cita el fabricante, el producto,
  el documento, la página y la sección de donde sale. Si el dato no está en las fichas, el sistema lo dice.</p>
  <div class="stats">
    <span><b>{len(rag.docs)}</b> FDS</span><span><b>{f"{len(rag.chunks):,}".replace(",", ".")}</b> fragmentos</span>
    <span><b>{len(FABS)}</b> fabricantes</span><span>granite-embedding + BM25</span><span>qwen2.5:3b · Ollama</span>
  </div>
</div>
"""

EMPTY = '<div class="empty">Escribe una pregunta o elige un ejemplo y pulsa <b>Consultar</b>. La respuesta tarda unos 30 segundos.</div>'


def _pages(s):
    return s["pagina"] if s.get("pagina_fin") in (None, s["pagina"]) else f"{s['pagina']}-{s['pagina_fin']}"


def render_answer(r):
    text = html.escape(r["answer"])
    text = re.sub(r"\[(\d+)\]", r'<sup class="c">\1</sup>', text)
    no = r.get("insufficient")
    pill = "Sin evidencia suficiente" if no else "Respuesta con citas"
    out = f'<div class="ans{" no" if no else ""}"><span class="pill">{pill}</span><div>{text}</div>'
    meta = []
    if r.get("filter"):
        meta.append("búsqueda restringida a " + ", ".join(html.escape(f) for f in r["filter"]))
    if r.get("usage"):
        u = r["usage"]
        meta.append(f"{u['wall_s']} s · {u['prompt_tokens']} tokens de entrada · top-k {r.get('top_k')}")
    if meta:
        out += f'<div class="meta">{" · ".join(meta)}</div>'
    if r.get("warnings"):
        out += f'<div class="warn">Aviso: {html.escape("; ".join(r["warnings"]))}</div>'
    return out + "</div>"


def render_sources(r):
    if not r["sources"]:
        return '<div class="empty">No se recuperó ningún fragmento.</div>'
    cards = []
    for s in r["sources"]:
        cit = '<div class="tag">citado en la respuesta</div>' if s["citado"] else ""
        cards.append(
            f'<div class="card{" cit" if s["citado"] else ""}"><span class="n">[{s["n"]}]</span>'
            f'<div><div class="t">{html.escape(str(s["fabricante"]))} · {html.escape(str(s["producto"]))}'
            f' (v{html.escape(str(s["version"] or "s/d"))})</div>'
            f'<div class="d">Página {_pages(s)} · Sección {s["seccion"]} {html.escape(str(s["seccion_nombre"]))}</div>'
            f'<div class="d">{html.escape(str(s["documento"]))}</div>{cit}</div>'
            f'<span class="s">similitud<br>{float(s["score"]):.3f}</span></div>')
    return '<div class="src">' + "".join(cards) + "</div>"


def ask(question, k, productos):
    if not question or not question.strip():
        return EMPTY, "", ""
    docs = [LABEL2DOC[p] for p in productos] if productos else None
    r = rag.answer(question.strip(), int(k), docs)
    ctx = "\n\n---\n\n".join(r["contexts"]) if r.get("contexts") else "_Sin fragmentos._"
    return render_answer(r), render_sources(r), ctx


with gr.Blocks(title="RAG · Fichas de Datos de Seguridad") as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            gr.HTML('<div class="secbar">Pregunta</div>')
            q = gr.Textbox(label="Pregunta", placeholder="Ej.: ¿Cuál es el número ONU del ...?", lines=3)
            with gr.Row():
                k = gr.Slider(1, 10, value=3, step=1, label="top-k (fragmentos recuperados)")
            prods = gr.Dropdown(PRODS, multiselect=True, label="Restringir a productos (opcional)")
            btn = gr.Button("Consultar", variant="primary", size="lg")
            gr.Examples(EXAMPLES, inputs=q, label="Preguntas de ejemplo")
        with gr.Column(scale=7):
            gr.HTML('<div class="secbar">Respuesta</div>')
            ans = gr.HTML(EMPTY)
            gr.HTML('<div class="secbar">Fuentes (trazabilidad)</div>')
            src = gr.HTML("")
            with gr.Accordion("Ver el texto de los fragmentos recuperados", open=False):
                ctx = gr.Markdown()
    gr.HTML('<div id="foot">Corre en AWS EC2 m7a.large (2 núcleos, 8 GB, sin GPU) con modelos open source en Ollama. '
            'Código: github.com/CamiloCometa/rag-fds</div>')
    btn.click(ask, [q, k, prods], [ans, src, ctx])
    q.submit(ask, [q, k, prods], [ans, src, ctx])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share="--share" in sys.argv,
                theme=gr.themes.Default(primary_hue="red", neutral_hue="slate"), css=CSS, head=HEAD)
