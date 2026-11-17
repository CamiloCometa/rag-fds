"""Genera 2 FDS sintéticas (ficticias) para probar el pipeline sin las fichas reales.

NO forman parte del corpus del parcial. Uso:
    python tests/make_synthetic_fds.py tests/fixtures/pdfs
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

SECTIONS = [
    "IDENTIFICACIÓN DE LA SUSTANCIA O LA MEZCLA Y DE LA SOCIEDAD O LA EMPRESA",
    "IDENTIFICACIÓN DE LOS PELIGROS",
    "COMPOSICIÓN/INFORMACIÓN SOBRE LOS COMPONENTES",
    "PRIMEROS AUXILIOS",
    "MEDIDAS DE LUCHA CONTRA INCENDIOS",
    "MEDIDAS EN CASO DE VERTIDO ACCIDENTAL",
    "MANIPULACIÓN Y ALMACENAMIENTO",
    "CONTROLES DE EXPOSICIÓN/PROTECCIÓN INDIVIDUAL",
    "PROPIEDADES FÍSICAS Y QUÍMICAS",
    "ESTABILIDAD Y REACTIVIDAD",
    "INFORMACIÓN TOXICOLÓGICA",
    "INFORMACIÓN ECOLÓGICA",
    "CONSIDERACIONES RELATIVAS A LA ELIMINACIÓN",
    "INFORMACIÓN RELATIVA AL TRANSPORTE",
    "INFORMACIÓN REGLAMENTARIA",
    "OTRA INFORMACIÓN",
]


def pictogram(path: Path):
    img = Image.new("RGB", (120, 120), "white")
    d = ImageDraw.Draw(img)
    d.polygon([(60, 4), (116, 60), (60, 116), (4, 60)], outline="red", width=6)
    d.text((48, 50), "!", fill="black")
    d.text((30, 70), "GHS07", fill="black")
    img.save(path)


def build(out: Path, fabricante: str, producto: str, version: str, fecha: str,
          omit: set[int], flash: str, solvent: tuple[str, str, str]):
    styles = getSampleStyleSheet()
    pic = out.parent / f"_{producto}_pict.png"
    pictogram(pic)

    def header(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(40, 815, f"{fabricante} - Ficha de Datos de Seguridad - {producto}")
        canvas.drawString(40, 805, f"Versión: {version}   Fecha de revisión: {fecha}")
        canvas.drawRightString(555, 20, f"Página {doc.page}")
        canvas.restoreState()

    story = [Paragraph(f"FICHA DE DATOS DE SEGURIDAD - {producto}", styles["Title"])]
    for i, title in enumerate(SECTIONS, start=1):
        if i in omit:
            continue
        story.append(Paragraph(f"SECCIÓN {i}: {title}", styles["Heading2"]))
        if i == 1:
            story.append(Paragraph(f"1.1 Identificador del producto. Nombre del producto: {producto}", styles["Normal"]))
            story.append(Paragraph(f"1.3 Proveedor: {fabricante} S.A.S., Bogotá, Colombia. Teléfono de emergencia: +57 601 555 0000", styles["Normal"]))
        elif i == 2:
            story.append(Paragraph("Clasificación: Líquido inflamable Cat. 3 (H226). Irritación cutánea Cat. 2 (H315).", styles["Normal"]))
            story.append(Paragraph("Pictogramas de peligro:", styles["Normal"]))
            story.append(RLImage(str(pic), width=60, height=60))
            story.append(Paragraph("Palabra de advertencia: Atención. Consejos de prudencia: P210, P280, P302+P352.", styles["Normal"]))
        elif i == 3:
            data = [["Componente", "N.º CAS", "Concentración (% p/p)", "Clasificación"],
                    [solvent[0], solvent[1], solvent[2], "Flam. Liq. 3; H226"],
                    ["Dióxido de titanio", "13463-67-7", "10 - 25", "Carc. 2; H351"],
                    ["Carbonato de calcio", "471-34-1", "5 - 10", "No clasificado"]]
            t = Table(data)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
            story.append(t)
        elif i == 8:
            story.append(Paragraph(f"Límite de exposición profesional ({solvent[0]}): TLV-TWA 100 ppm (ACGIH).", styles["Normal"]))
            story.append(Paragraph("Protección respiratoria: respirador con filtro para vapores orgánicos tipo A. Guantes de nitrilo (espesor ≥ 0,4 mm).", styles["Normal"]))
        elif i == 9:
            data = [["Propiedad", "Valor", "Método"],
                    ["Estado físico", "Líquido viscoso", "-"],
                    ["Punto de inflamación", flash, "ASTM D93"],
                    ["Densidad relativa (25 °C)", "1,32 g/cm³", "ASTM D1475"],
                    ["pH", "No aplica", "-"],
                    ["COV", "380 g/L", "EPA 24"]]
            t = Table(data)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
            story.append(t)
            story.append(PageBreak())
        else:
            story.append(Paragraph(f"Texto de prueba de la sección {i} para {producto}. "
                                   "Mantener alejado de fuentes de ignición. " * 3, styles["Normal"]))
        story.append(Spacer(1, 8))
    SimpleDocTemplate(str(out), pagesize=A4, topMargin=60).build(story, onFirstPage=header, onLaterPages=header)
    pic.unlink()


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/pdfs")
    (root / "FDS 2023").mkdir(parents=True, exist_ok=True)
    (root / "FDS 2024").mkdir(parents=True, exist_ok=True)
    build(root / "FDS 2024" / "FDS 6 - ESMALTE SINTETICO BLANCO.pdf", "Pinturas Andinas",
          "Esmalte Sintético Blanco", "3.1", "15/02/2024", omit=set(), flash="38 °C",
          solvent=("Varsol (nafta de petróleo)", "64742-88-7", "20 - 30"))
    build(root / "FDS 2023" / "FDS 2 - VINILO TIPO 1 - COLORSUR.pdf", "ColorSur",
          "Vinilo Tipo 1 Blanco", "2.0", "03/07/2023", omit={12}, flash="No inflamable (> 100 °C)",
          solvent=("Etilenglicol", "107-21-1", "1 - 3"))
    print("ok", root)
