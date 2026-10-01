"""
Dibujado del panel informativo sobre el video anotado.

Las fuentes Hershey de OpenCV no tienen acentos: cv2.putText("Agitacion")
funciona, cv2.putText("Agitacion" con tilde) sale con basura. Por eso el texto
se rasteriza con Pillow (que ya viene con torchvision) usando una fuente del
sistema, y se cae de vuelta a OpenCV sin acentos si no hay ninguna disponible.
"""

from __future__ import annotations

import unicodedata

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont

    _HAY_PIL = True
except ImportError:
    _HAY_PIL = False

_CANDIDATAS = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
_CANDIDATAS_NEGRITA = [
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _sin_acentos(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


class Panel:
    """Acumula textos y los pinta de una sola pasada (una conversion PIL por cuadro)."""

    def __init__(self):
        self.fuentes: dict[tuple[int, bool], object] = {}
        self.ok = _HAY_PIL and self._cargar(15, False) is not None

    def _cargar(self, tam: int, negrita: bool):
        clave = (tam, negrita)
        if clave in self.fuentes:
            return self.fuentes[clave]
        fuente = None
        if _HAY_PIL:
            for ruta in (_CANDIDATAS_NEGRITA if negrita else _CANDIDATAS):
                try:
                    fuente = ImageFont.truetype(ruta, tam)
                    break
                except Exception:
                    continue
        self.fuentes[clave] = fuente
        return fuente

    def escribir(self, img, textos):
        """textos: lista de (cadena, (x, y), color_bgr, tamano, negrita)."""
        if not textos:
            return img
        if not self.ok:
            for s, (x, y), color, tam, negrita in textos:
                escala = tam / 30.0
                grosor = 2 if negrita else 1
                plano = _sin_acentos(s)
                cv2.putText(img, plano, (x, y + tam), cv2.FONT_HERSHEY_SIMPLEX,
                            escala, (0, 0, 0), grosor + 2, cv2.LINE_AA)
                cv2.putText(img, plano, (x, y + tam), cv2.FONT_HERSHEY_SIMPLEX,
                            escala, color, grosor, cv2.LINE_AA)
            return img

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        lienzo = ImageDraw.Draw(pil)
        for s, (x, y), color, tam, negrita in textos:
            fuente = self._cargar(tam, negrita)
            rgb = (color[2], color[1], color[0])
            lienzo.text((x, y), s, font=fuente, fill=rgb,
                        stroke_width=2, stroke_fill=(0, 0, 0))
        img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return img


def banda(img, y0, y1, alpha=0.65, color=(0, 0, 0)):
    """Franja semitransparente, para que el texto se lea sobre cualquier escena."""
    recorte = img[y0:y1]
    if recorte.size == 0:
        return
    capa = np.full_like(recorte, color, dtype=np.uint8)
    img[y0:y1] = cv2.addWeighted(recorte, 1 - alpha, capa, alpha, 0)


def rect_punteado(img, p1, p2, color, grosor=1, guion=10, hueco=7):
    """Rectangulo a trazos: se usa para la ZONA que analiza el modelo de
    violencia, que no es una deteccion sino un area de analisis."""
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, guion + hueco):
        cv2.line(img, (x, y1), (min(x + guion, x2), y1), color, grosor)
        cv2.line(img, (x, y2), (min(x + guion, x2), y2), color, grosor)
    for y in range(y1, y2, guion + hueco):
        cv2.line(img, (x1, y), (x1, min(y + guion, y2)), color, grosor)
        cv2.line(img, (x2, y), (x2, min(y + guion, y2)), color, grosor)
