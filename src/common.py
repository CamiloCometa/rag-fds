"""Utilidades compartidas: configuración, rutas, normalización de texto y secciones FDS."""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Nombres canónicos de las 16 secciones (Res. 0773/2021 Colombia, SGA/GHS rev., Reglamento REACH anexo II)
SECCIONES = {
    1: "Identificación del producto y de la empresa",
    2: "Identificación de los peligros",
    3: "Composición / información sobre los componentes",
    4: "Primeros auxilios",
    5: "Medidas de lucha contra incendios",
    6: "Medidas en caso de vertido / liberación accidental",
    7: "Manipulación y almacenamiento",
    8: "Controles de exposición / protección personal",
    9: "Propiedades físicas y químicas",
    10: "Estabilidad y reactividad",
    11: "Información toxicológica",
    12: "Información ecológica",
    13: "Consideraciones relativas a la eliminación",
    14: "Información relativa al transporte",
    15: "Información reglamentaria",
    16: "Otra información",
}

# Palabras clave (sin tildes, minúsculas) para reconocer cada sección aunque el
# fabricante use otra redacción, el inglés, o el formato antiguo NTC 4435/ANSI
# (donde 2 = Composición y 3 = Identificación de riesgos).
KEYWORDS = {
    1: ["identificacion", "identification", "producto y de la", "sustancia", "empresa", "proveedor", "fabricante"],
    2: ["peligro", "riesgo", "hazard"],
    3: ["composicion", "componentes", "ingredientes", "composition", "ingredients"],
    4: ["primeros auxilios", "auxilio", "first aid"],
    5: ["incendio", "fire", "extincion"],
    6: ["vertido", "derrame", "liberacion", "liberaciones", "fuga", "accidental", "escape"],
    7: ["manipulacion", "almacenamiento", "almacenaje", "handling", "storage", "manejo"],
    8: ["exposicion", "proteccion", "exposure", "protection", "epp"],
    9: ["propiedades fisicas", "fisicas y quimicas", "physical", "propiedades"],
    10: ["estabilidad", "reactividad", "stability", "reactivity"],
    11: ["toxicolog"],
    12: ["ecolog", "ecotox", "medio ambiente", "ambiental", "ecological"],
    13: ["eliminacion", "disposicion", "desecho", "disposal", "residuos"],
    14: ["transporte", "transport"],
    15: ["reglament", "regulatoria", "normativ", "regulatory", "legislac"],
    16: ["otra informacion", "otras informaciones", "other information", "informacion adicional", "otros datos"],
}
# Palabras específicas (sin las genéricas "identificación", "información", "propiedades", "protección")
SPECIFIC = {
    1: ["producto y", "del producto", "sustancia", "substancia", "empresa", "proveedor", "fabricante", "compania",
        "sociedad", "of the product"],
    2: ["peligro", "riesgo", "hazard"],
    3: ["composicion", "componentes", "ingredientes", "composition", "ingredients"],
    4: ["primeros auxilios", "auxilio", "first aid"],
    5: ["incendio", "fire", "extincion"],
    6: ["vertido", "derrame", "liberacion", "liberaciones", "fuga", "accidental", "escape"],
    7: ["manipulacion", "almacenamiento", "almacenaje", "handling", "storage", "manejo"],
    8: ["exposicion", "proteccion personal", "proteccion individual", "exposure", "epp"],
    9: ["fisicas", "physical"],
    10: ["estabilidad", "reactividad", "stability", "reactivity"],
    11: ["toxicolog"],
    12: ["ecolog", "ecotox", "medio ambiente", "ambiental"],
    13: ["eliminacion", "disposicion", "desecho", "disposal", "residuos"],
    14: ["transporte", "transport"],
    15: ["reglament", "regulatoria", "normativ", "regulatory", "legislac"],
    16: ["otra informacion", "otras informaciones", "other information", "informacion adicional", "otros datos"],
}

# Encabezado de sección: "SECCIÓN 3:", "Sección 3 -", "EPÍGRAFE 3.", "3. COMPOSICIÓN", "## **3 IDENTIF...**"
HEADING_RE = re.compile(
    r"^[\s#>*_|-]*(?P<pref>secci[oó]n|section|ep[ií]grafe|cap[ií]tulo)?\s*(?P<num>1[0-6]|0?[1-9])(?!\s*[.,]\s*\d)(?!\d)"
    r"\s*[.:)\-–—]?\s*(?P<title>[^\n]{0,260})$",
    re.IGNORECASE,
)


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get("RAG_FDS_CONFIG") or ROOT / "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = str((ROOT / v).resolve())
    return cfg


def config_hash(cfg: dict, keys=("chunking", "embeddings", "generator", "retrieval")) -> str:
    sub = {k: cfg.get(k) for k in keys}
    return hashlib.sha1(json.dumps(sub, sort_keys=True).encode()).hexdigest()[:10]


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm(s: str) -> str:
    s = strip_accents(s or "").lower()
    s = re.sub(r"[*_`#|]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", norm(s)).strip("_")
    return s[:60]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


_NOT_TITLE_START = {"la", "el", "los", "las", "y", "para", "en", "de", "del", "a", "sobre", "con", "se", "que",
                    "por", "o", "e", "es", "son", "no", "si"}
_REF_WORDS = {"la", "las", "ver", "vea", "consulte", "consultar", "vease", "en", "a", "de", "y", "segun", "see"}
_HEAD_START = {"identificacion", "informacion", "composicion", "primeros", "medidas", "manipulacion", "manejo",
               "controles", "control", "propiedades", "estabilidad", "consideraciones", "otra", "otras", "combate",
               "liberaciones", "hazard", "hazards", "first", "fire", "accidental", "handling", "exposure", "physical",
               "stability", "toxicological", "ecological", "disposal", "transport", "regulatory", "other"}


def canonical_section(num: int, t: str):
    """Sección canónica (1-16) según el contenido del título; respeta el número si es coherente."""
    t = t[:120]
    scores = {k: sum(w in t for w in ws) for k, ws in SPECIFIC.items()}
    if scores.get(num):
        return num
    best = max(scores.values())
    if best:
        top = [k for k, v in scores.items() if v == best]
        if len(top) == 1:
            return top[0]
    if any(k in t for k in KEYWORDS[num]):
        return num
    return None


def _heading_like(s: str, t: str, canon: int) -> bool:
    """Sin prefijo "SECCIÓN": exige formato de título (markdown/negrita, mayúsculas) o que el
    título EMPIECE como un encabezado (evita listas numeradas como "8. Use protección…")."""
    if s.lstrip("|- ").startswith(("#", "**")):
        return True
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        return True
    first = (t.split() or [""])[0]
    return len(s) <= 90 and (any(t.startswith(k) for k in KEYWORDS[canon]) or first in _HEAD_START)


MIDLINE_RE = re.compile(r"(?:secci[oó]n|section|ep[ií]grafe)\s*(?P<num>1[0-6]|0?[1-9])(?!\s*[.,]\s*\d)(?!\d)\s*[.:)\-–—]?\s*(?P<title>.{0,120})",
                        re.IGNORECASE)


def _clean_title(x: str) -> str:
    return re.sub(r"<[^>]+>|[*_#|~]+", " ", x).strip(" :.-")[:150]


def _match_segment(s: str):
    s = s.strip()
    if not s:
        return None
    m = HEADING_RE.match(s[:260])
    if m:
        num = int(m.group("num"))
        title = m.group("title") or ""
        t = norm(re.sub(r"<[^>]+>", " ", title))
        has_pref = bool(m.group("pref"))
        canon = canonical_section(num, t)
        first = (t.split() or [""])[0]
        if first in _NOT_TITLE_START:  # "Sección 8 la información adicional…" (referencia partida en dos líneas)
            return None
        if has_pref and canon and (canon == num or _heading_like(s, t, canon)):
            return num, _clean_title(title), canon
        if has_pref and len(t) < 3 and len(s) <= 40:  # "SECCIÓN 8" con el título en la línea siguiente
            return num, "", num
        if not has_pref and canon and len(s) <= 220 and len(t) >= 4 and _heading_like(s, t, canon):
            return num, _clean_title(title), canon
    return None


BOLD_NUM_RE = re.compile(r"(?:\*\*|#)\s*(?P<num>1[0-6]|0?[1-9])\s*[.:)\-–—]\s*(?P<title>[A-ZÁÉÍÓÚÑ][^*\n]{3,140})")


def _midline(s: str):
    """Encabezado incrustado en una línea más larga (título fusionado, <mark>, negrita, pie de página)."""
    for mm in MIDLINE_RE.finditer(s):
        before = re.findall(r"\w+", norm(s[:mm.start()]))
        if before and before[-1] in _REF_WORDS:
            continue
        num = int(mm.group("num"))
        raw_title = re.sub(r"<[^>]+>", " ", mm.group("title"))
        t = norm(raw_title)
        canon = canonical_section(num, t)
        first = (t.split() or [""])[0]
        if not canon or not (first in _HEAD_START or any(t.startswith(k) for k in SPECIFIC[canon])):
            continue
        ctx = s[max(0, mm.start() - 8):mm.start()]
        letters = [c for c in raw_title[:30] if c.isalpha()]
        upper = letters and sum(c.isupper() for c in letters) / len(letters) > 0.6
        formatted = any(tag in ctx for tag in ("<mark>", "**", "#", ">")) or mm.start() == 0
        if upper or formatted or s.startswith(("#", "**", "|**", "| **")):
            return num, _clean_title(mm.group("title")), canon
    for mm in BOLD_NUM_RE.finditer(s):
        num = int(mm.group("num"))
        t = norm(mm.group("title"))
        canon = canonical_section(num, t)
        letters = [c for c in mm.group("title")[:30] if c.isalpha()]
        if canon and letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
            return num, _clean_title(mm.group("title")), canon
    return None


def match_heading(line: str):
    """(número en el documento, título, sección canónica) si la línea es un encabezado de sección.

    Divide en <br> (texto de tablas/imágenes), acepta "SECCIÓN N" + título coherente, números
    sin prefijo solo con formato de título, y "SECCIÓN N" dentro de una línea con formato de
    encabezado (título fusionado). La sección canónica sale del contenido del título, de modo
    que el formato antiguo (2 = Composición, 3 = Riesgos) se mapea correctamente.
    """
    line = line or ""
    for seg in re.split(r"<br\s*/?>", line):
        r = _match_segment(seg)
        if r:
            return r
    if "|" in line:  # encabezado partido en celdas de tabla: "|**Sección 1. Identifi**<br>|**cación**|"
        joined = re.sub(r"\*\*(?:<br>)?\|\*\*", "", line)
        joined = re.sub(r"<br\s*/?>", " ", joined).replace("|", " ")
        r = _match_segment(joined) or _midline(joined)
        if r:
            return r
    return _midline(line)


def select_sections(cands):
    """Elige la secuencia más larga de encabezados con número de documento creciente
    (descarta referencias y falsos positivos) y, para cada sección, su primera aparición válida.

    cands: lista de (idx, num_doc, canon, title, page) en orden de aparición.
    Devuelve {canon: {num, numero_en_documento, titulo_en_documento, pagina, linea}}.
    """
    n = len(cands)
    if not n:
        return {}
    best, prev = [1] * n, [-1] * n
    for i in range(n):
        for j in range(i):
            if cands[j][1] < cands[i][1] and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    i = max(range(n), key=lambda k: (best[k], -k))
    chain = []
    while i != -1:
        chain.append(i)
        i = prev[i]
    chain.reverse()
    # adelantar cada elegido a su primera aparición entre el elegido anterior y el siguiente
    out = {}
    for pos, ci in enumerate(chain):
        lo = cands[chain[pos - 1]][0] if pos else -1
        num = cands[ci][1]
        first = next(c for c in cands if c[1] == num and lo < c[0] <= cands[ci][0])
        idx, num_doc, canon, title, page = first
        if canon not in out:
            out[canon] = {"num": canon, "numero_en_documento": num_doc, "titulo_en_documento": title,
                          "pagina": page, "linea": idx}
    return out


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
