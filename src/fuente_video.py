"""
Fuente de video con decodificacion UNICA y buffer circular.

Todos los consumidores (personas, armas, violencia, evidencia) se alimentan de
este mismo bucle. Puntos clave:

  - Muestreo por TIMESTAMP, no por indice de frame: tolera fuentes a 25 fps,
    29.97 fps o con fps variable.
  - grab() en todos los cuadros, retrieve() solo en los que se procesan.
    H.264 encadena los cuadros, asi que no se puede saltar el grab; pero la
    conversion a BGR (que es lo caro) solo se paga en los que se usan.
    Medido: 1.76 ms grab vs 6.53 ms conversion.
  - El buffer guarda los cuadros REESCALADOS. A 1080p sin reescalar, 20 s de
    buffer serian ~1.2 GB de RAM; a 640 de ancho son ~138 MB.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Muestra:
    t: float                 # segundos desde el inicio del video
    completo: np.ndarray     # cuadro a resolucion nativa (para los detectores)
    pequeno: np.ndarray      # cuadro reescalado (buffer, violencia, anotacion)


class FuenteVideo:
    def __init__(
        self,
        ruta: str | Path,
        fps_base: float,
        ancho_buffer: int,
        buffer_segundos: float,
        inicio: float = 0.0,
        fin: float | None = None,
    ):
        self.ruta = str(ruta)
        self.cap = cv2.VideoCapture(self.ruta)
        if not self.cap.isOpened():
            # RuntimeError y no SystemExit: si un video del lote falla, el resto
            # debe seguir procesandose.
            raise RuntimeError(f"No se pudo abrir el video: {self.ruta}")

        self.fps_nativo = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.ancho = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.alto = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        self.duracion = total / self.fps_nativo if total else None

        self.fps_base = float(fps_base)
        self.periodo = 1.0 / self.fps_base
        self.inicio = float(inicio)
        self.fin = fin

        # escala del buffer
        self.escala = min(1.0, ancho_buffer / max(self.ancho, 1))
        self.ancho_b = int(round(self.ancho * self.escala))
        self.alto_b = int(round(self.alto * self.escala))

        self.buffer: deque[tuple[float, np.ndarray]] = deque(
            maxlen=max(4, int(buffer_segundos * self.fps_base))
        )
        self.t_actual = 0.0

    # ------------------------------------------------------------------ util
    def a_buffer(self, xyxy) -> tuple[int, int, int, int]:
        """Convierte coordenadas del cuadro completo a coordenadas del buffer."""
        x1, y1, x2, y2 = xyxy
        e = self.escala
        return int(x1 * e), int(y1 * e), int(x2 * e), int(y2 * e)

    # ------------------------------------------------------------- iteracion
    def __iter__(self):
        idx = 0
        siguiente = self.inicio
        while True:
            if not self.cap.grab():
                break
            t = idx / self.fps_nativo
            idx += 1

            if self.fin is not None and t > self.fin:
                break
            if t + 1e-9 < siguiente:
                continue

            ok, cuadro = self.cap.retrieve()
            if not ok:
                break
            siguiente = max(t, siguiente) + self.periodo

            pequeno = (
                cuadro
                if self.escala >= 1.0
                else cv2.resize(cuadro, (self.ancho_b, self.alto_b), interpolation=cv2.INTER_AREA)
            )
            self.buffer.append((t, pequeno))
            self.t_actual = t
            yield Muestra(t=t, completo=cuadro, pequeno=pequeno)

        self.cap.release()

    # ---------------------------------------------------------------- buffer
    def rango(self, t0: float, t1: float) -> list[np.ndarray]:
        """Cuadros del buffer dentro de [t0, t1], en orden."""
        return [f for t, f in self.buffer if t0 <= t <= t1]

    def muestrear(self, t_fin: float, n: int, paso_seg: float) -> list[np.ndarray] | None:
        """
        n cuadros equiespaciados cada `paso_seg`, terminando en t_fin.
        Devuelve None si el buffer todavia no cubre la ventana completa.

        Reproduce el muestreo con el que se entreno MC3-18: 16 cuadros
        repartidos sobre ~4.8 s.
        """
        if not self.buffer:
            return None
        t_ini = t_fin - (n - 1) * paso_seg
        if self.buffer[0][0] > t_ini + 1e-6:
            return None

        tiempos = np.array([t for t, _ in self.buffer])
        salida = []
        for k in range(n):
            objetivo = t_ini + k * paso_seg
            i = int(np.argmin(np.abs(tiempos - objetivo)))
            salida.append(self.buffer[i][1])
        return salida
