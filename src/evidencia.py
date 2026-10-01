"""
Clip de evidencia a partir del buffer circular.

Ningun repositorio de origen tenia esto, y sin ello no se puede rellenar
`evidence_url` (RESUMEN seccion 13).

El clip abarca [t_evento - pre, t_evento + post]. La parte posterior todavia no
existe cuando se dispara el evento, asi que la escritura queda PENDIENTE y se
resuelve cuando el reloj del video la alcanza.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class Pendiente:
    event_id: str
    t_evento: float
    t_desde: float
    t_hasta: float
    destino: Path


class Evidencia:
    def __init__(self, cfg, fuente, carpeta: Path):
        self.fuente = fuente
        self.carpeta = Path(carpeta)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.pre = float(cfg.evidencia.pre_segundos)
        self.post = float(cfg.evidencia.post_segundos)
        self.fps = float(cfg.evidencia.fps_clip)
        self.pendientes: list[Pendiente] = []
        self.escritos: list[Path] = []

    def solicitar(self, evento) -> Path:
        destino = self.carpeta / f"{evento.base}.mp4"
        self.pendientes.append(
            Pendiente(
                event_id=evento.event_id,
                t_evento=evento.t_video,
                t_desde=max(0.0, evento.t_video - self.pre),
                t_hasta=evento.t_video + self.post,
                destino=destino,
            )
        )
        return destino

    def procesar(self, t: float) -> None:
        listos = [p for p in self.pendientes if t >= p.t_hasta]
        for p in listos:
            self._escribir(p)
            self.pendientes.remove(p)

    def cerrar(self) -> None:
        """Al terminar el video se vuelca lo que haya, aunque falte el post."""
        for p in list(self.pendientes):
            self._escribir(p)
        self.pendientes.clear()

    def _escribir(self, p: Pendiente) -> None:
        cuadros = self.fuente.rango(p.t_desde, p.t_hasta)
        if not cuadros:
            return
        alto, ancho = cuadros[0].shape[:2]
        escritor = cv2.VideoWriter(
            str(p.destino), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (ancho, alto)
        )
        for c in cuadros:
            escritor.write(c)
        escritor.release()
        self.escritos.append(p.destino)
