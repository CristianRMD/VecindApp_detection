"""
Senales de postura a partir de los 17 puntos clave de YOLOv8n-pose.

POR QUE EXISTE
--------------
El detector de armas falla porque una pistola mide 20-40 px sobre video CCTV y
solo se detecta en el 14-29% de los cuadros. Pero **la postura de apuntar mide
cientos de pixeles**. Se puede reconocer el ACTO de apuntar sin llegar a ver el
arma, que es exactamente lo que se pierde en Video3 y Video5.

Y cubre los indicadores de RESUMEN seccion 2 que ningun modelo actual captura:
"intimidacion fisica o sometimiento".

  apuntar       brazo extendido, separado del cuerpo, y cuya prolongacion pasa
                por el torso de OTRA persona. Esa ultima condicion es la que lo
                hace especifico: estirarse hacia un estante no apunta a nadie.
  manos_arriba  ambas munecas por encima de sus hombros -> sometimiento.
  en_suelo      torso mas horizontal que vertical -> caida, persona reducida.

La combinacion `apuntar` + `manos_arriba` sostenidas a la vez es, visualmente,
un atraco a mano armada aunque el arma nunca se vea.

Cuesta casi nada: yolov8n-pose son 23.3 ms contra 21.4 ms del detector normal,
y REEMPLAZA a ese detector en vez de sumarse a el.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Indices COCO de los 17 puntos clave
NARIZ = 0
HOMBRO_I, HOMBRO_D = 5, 6
CODO_I, CODO_D = 7, 8
MUNECA_I, MUNECA_D = 9, 10
CADERA_I, CADERA_D = 11, 12
RODILLA_I, RODILLA_D = 13, 14
TOBILLO_I, TOBILLO_D = 15, 16

BRAZOS = ((HOMBRO_I, CODO_I, MUNECA_I), (HOMBRO_D, CODO_D, MUNECA_D))

# Pares para dibujar el esqueleto
ESQUELETO = [
    (HOMBRO_I, HOMBRO_D), (HOMBRO_I, CODO_I), (CODO_I, MUNECA_I),
    (HOMBRO_D, CODO_D), (CODO_D, MUNECA_D),
    (HOMBRO_I, CADERA_I), (HOMBRO_D, CADERA_D), (CADERA_I, CADERA_D),
    (CADERA_I, RODILLA_I), (RODILLA_I, TOBILLO_I),
    (CADERA_D, RODILLA_D), (RODILLA_D, TOBILLO_D),
]


@dataclass
class Postura:
    canon: int
    manos_arriba: bool = False
    n_manos: int = 0
    en_suelo: bool = False
    brazo: tuple | None = None          # (hombro, muneca) del brazo extendido
    apunta_a: list[int] = field(default_factory=list)


def _pt(kpts, i: int, conf_min: float):
    if kpts is None or i >= len(kpts):
        return None
    x, y, c = kpts[i][:3]
    return (float(x), float(y)) if float(c) >= conf_min else None


def _medio(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


class AnalizadorPose:
    def __init__(self, cfg):
        c = cfg.get("pose", {})
        self.activo = bool(c.get("activo", True))
        self.conf_kp = float(c.get("conf_keypoint", 0.5))
        self.ext_min = float(c.get("extension_min", 0.88))
        self.alejamiento = float(c.get("alejamiento_factor", 1.15))
        self.radio = float(c.get("radio_apuntado", 0.9))
        self.alcance = float(c.get("alcance_max_factor", 3.0))
        self.margen_manos = float(c.get("margen_manos", 0.10))
        self.manos_min = int(c.get("manos_min", 1))
        self.sostenido = float(c.get("sostenido_min_seg", 0.8))

        self.posturas: list[Postura] = []
        self.desde: dict[str, float | None] = {
            "apuntar": None, "manos_arriba": None, "en_suelo": None
        }

    # ------------------------------------------------------------ interno --
    def _marcar(self, nombre: str, encendida: bool, t: float) -> None:
        if encendida:
            if self.desde[nombre] is None:
                self.desde[nombre] = t
        else:
            self.desde[nombre] = None

    def _sostenida(self, nombre: str, t: float) -> bool:
        ini = self.desde[nombre]
        return ini is not None and (t - ini) >= self.sostenido

    def _torso(self, tk):
        """Centro del torso: caderas si se ven, si no el centro de la caja."""
        caderas = _medio(_pt(tk.kpts, CADERA_I, self.conf_kp),
                         _pt(tk.kpts, CADERA_D, self.conf_kp))
        hombros = _medio(_pt(tk.kpts, HOMBRO_I, self.conf_kp),
                         _pt(tk.kpts, HOMBRO_D, self.conf_kp))
        m = _medio(caderas, hombros)
        return m if m is not None else tk.centro

    def _ancho_hombros(self, tk) -> float:
        hi = _pt(tk.kpts, HOMBRO_I, self.conf_kp)
        hd = _pt(tk.kpts, HOMBRO_D, self.conf_kp)
        if hi and hd:
            return max(math.dist(hi, hd), 1.0)
        return max(tk.ancho * 0.45, 1.0)   # respaldo razonable

    # --------------------------------------------------------- actualizar --
    def actualizar(self, tracks, t: float) -> list[Postura]:
        if not self.activo:
            self.posturas = []
            return []

        posturas = []
        for tk in tracks:
            p = Postura(canon=tk.canon)
            k = tk.kpts
            if k is None:
                posturas.append(p)
                continue

            # --- torso horizontal (en el suelo) ---
            hombros = _medio(_pt(k, HOMBRO_I, self.conf_kp), _pt(k, HOMBRO_D, self.conf_kp))
            caderas = _medio(_pt(k, CADERA_I, self.conf_kp), _pt(k, CADERA_D, self.conf_kp))
            if hombros and caderas:
                dx, dy = abs(hombros[0] - caderas[0]), abs(hombros[1] - caderas[1])
                largo = math.hypot(dx, dy)
                if largo > 0.15 * max(tk.alto, 1.0):
                    p.en_suelo = dy < 0.6 * dx

            # --- brazo extendido (se resuelve ANTES que las manos) ---
            ancho_h = self._ancho_hombros(tk)
            torso = self._torso(tk)
            brazo_extendido = None
            for idx, (hombro, codo, muneca) in enumerate(BRAZOS):
                s, e, w = (_pt(k, hombro, self.conf_kp),
                           _pt(k, codo, self.conf_kp),
                           _pt(k, muneca, self.conf_kp))
                if not (s and e and w):
                    continue
                l = math.dist(s, e) + math.dist(e, w)
                if l < 1e-6:
                    continue
                # extension = 1.0 cuando el brazo esta completamente recto.
                # Por si solo discrimina poco (el 34-73% de los brazos la pasan):
                # quien filtra de verdad son el alejamiento y, sobre todo, que la
                # prolongacion del brazo pase por OTRA persona.
                if math.dist(s, w) / l < self.ext_min:
                    continue
                if math.dist(w, torso) < self.alejamiento * ancho_h:
                    continue
                p.brazo = (s, w)
                brazo_extendido = idx
                break

            # --- manos arriba (sometimiento) ---
            # Se EXCLUYE el brazo extendido: el que apunta suele llevar el arma
            # a la altura del hombro, y sin esto se le contaria como "manos
            # arriba" -- justo lo contrario de lo que significa.
            margen = self.margen_manos * max(tk.alto, 1.0)
            for idx, (muneca, hombro) in enumerate(((MUNECA_I, HOMBRO_I), (MUNECA_D, HOMBRO_D))):
                if idx == brazo_extendido:
                    continue
                w, s = _pt(k, muneca, self.conf_kp), _pt(k, hombro, self.conf_kp)
                if w and s and (s[1] - w[1]) > margen:   # y crece hacia abajo
                    p.n_manos += 1
            p.manos_arriba = p.n_manos >= self.manos_min

            posturas.append(p)

        # --- a quien apunta cada brazo extendido ---
        por_canon = {tk.canon: tk for tk in tracks}
        for p in posturas:
            if p.brazo is None:
                continue
            s, w = p.brazo
            dx, dy = w[0] - s[0], w[1] - s[1]
            norma = math.hypot(dx, dy)
            if norma < 1e-6:
                continue
            ux, uy = dx / norma, dy / norma
            tk = por_canon[p.canon]
            alcance = self.alcance * max(tk.alto, 1.0)

            for otro in tracks:
                if otro.canon == p.canon:
                    continue
                cx, cy = self._torso(otro)
                vx, vy = cx - w[0], cy - w[1]
                largo = vx * ux + vy * uy          # proyeccion sobre el rayo
                if largo <= 0 or largo > alcance:
                    continue                        # detras, o demasiado lejos
                perp = abs(vx * uy - vy * ux)       # distancia perpendicular
                if perp < self.radio * max(otro.ancho, 1.0):
                    p.apunta_a.append(otro.canon)

        self.posturas = posturas
        self._marcar("apuntar", any(p.apunta_a for p in posturas), t)
        self._marcar("manos_arriba", any(p.manos_arriba for p in posturas), t)
        self._marcar("en_suelo", any(p.en_suelo for p in posturas), t)
        return posturas

    # ------------------------------------------------------------ consulta --
    def senales(self, t: float) -> dict:
        apuntar = self._sostenida("apuntar", t)
        manos = self._sostenida("manos_arriba", t)
        suelo = self._sostenida("en_suelo", t)
        return {
            "apuntar": apuntar,
            "manos_arriba": manos,
            "en_suelo": suelo,
            # Uno apuntando y otro con las manos arriba, a la vez y sostenido:
            # eso ES un atraco a mano armada aunque el arma no se vea.
            "amenaza": bool(apuntar and manos),
            "n_apuntando": sum(1 for p in self.posturas if p.apunta_a),
            "n_manos_arriba": sum(1 for p in self.posturas if p.manos_arriba),
            "n_activas": int(apuntar) + int(manos) + int(suelo),
        }

    def reiniciar(self) -> None:
        self.posturas = []
        for k in self.desde:
            self.desde[k] = None
