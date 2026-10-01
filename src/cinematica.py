"""
Senales cinematicas derivadas de los tracks. Sin red neuronal, coste ~0.

Existe porque MC3-18 fue entrenado para 'fight' vs 'noFight', mientras que
RESUMEN seccion 2 incluye "intimidacion fisica o sometimiento". Un atraco tipico
de bodega (arma + obediencia, sin pelea) sale noFight. El propio informe de
validacion reporta "arresto policial -> noFight" como acierto, cuando para
VecindApp es un falso negativo.

Tres senales, todas escala-invariantes (normalizadas por el tamano de la caja,
para que no dependan de la distancia a la camara):

  proximidad : dos personas con los centros mas cerca que un factor del ancho
               medio de sus cajas, sostenido. -> forcejeo, sometimiento.
  agitacion  : velocidad del centro por encima del percentil de la linea base
               observada, con un piso fijo. -> golpes, empujones.
  aspecto    : caida brusca de la relacion alto/ancho de una caja respecto a su
               propia mediana. -> caida, agacharse, manos arriba.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np


class Cinematica:
    def __init__(self, cfg):
        c = cfg.cinematica
        self.activo = bool(c.activo)
        self.factor_prox = float(c.proximidad_factor)
        self.prox_min = float(c.proximidad_min_seg)
        self.percentil = float(c.percentil_agitacion)
        self.vel_min = float(c.umbral_velocidad_min)
        self.aspecto_delta = float(c.aspecto_delta)
        self.sostenido_min = float(c.sostenido_min_seg)

        self.baseline_min = int(c.get("muestras_baseline_min", 1000))
        self.baseline: deque[float] = deque(maxlen=int(c.ventana_baseline))
        self.previo: dict[int, tuple[float, tuple[float, float], float]] = {}
        self.aspectos: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=40))

        # instante desde el que cada senal esta continuamente activa
        self.desde: dict[str, float | None] = {"proximidad": None, "agitacion": None, "aspecto": None}
        self.valores: dict[str, float] = {"vel_max": 0.0, "dist_min": float("inf"), "aspecto_min": 1.0}

    # ------------------------------------------------------------ interno --
    def _umbral_velocidad(self) -> float:
        # Con poca linea base manda el piso. En un clip corto que es todo evento
        # no hay comportamiento "normal" del cual aprender, y el percentil
        # adaptativo subiria hasta anularse a si mismo.
        if len(self.baseline) < self.baseline_min:
            return self.vel_min
        return max(self.vel_min, float(np.percentile(self.baseline, self.percentil)))

    def _marcar(self, nombre: str, encendida: bool, t: float) -> None:
        if encendida:
            if self.desde[nombre] is None:
                self.desde[nombre] = t
        else:
            self.desde[nombre] = None

    def _sostenida(self, nombre: str, t: float, minimo: float) -> bool:
        ini = self.desde[nombre]
        return ini is not None and (t - ini) >= minimo

    # --------------------------------------------------------- actualizar --
    def actualizar(self, tracks, t: float) -> None:
        if not self.activo:
            return

        # --- agitacion + aspecto (por track) ---
        vel_max = 0.0
        aspecto_bajo = False
        aspecto_min = 1.0
        for tk in tracks:
            alto = max(tk.alto, 1.0)
            ant = self.previo.get(tk.canon)
            if ant is not None:
                t_ant, c_ant, alto_ant = ant
                dt = t - t_ant
                if dt > 1e-3:
                    ref = max((alto + alto_ant) / 2.0, 1.0)
                    v = math.dist(tk.centro, c_ant) / (ref * dt)  # alturas-de-caja / s
                    vel_max = max(vel_max, v)
                    self.baseline.append(v)
            self.previo[tk.canon] = (t, tk.centro, alto)

            rel = alto / max(tk.ancho, 1.0)
            hist = self.aspectos[tk.canon]
            if len(hist) >= 8:
                mediana = float(np.median(hist))
                if mediana > 0 and rel < mediana * (1.0 - self.aspecto_delta):
                    aspecto_bajo = True
                    aspecto_min = min(aspecto_min, rel / mediana)
            hist.append(rel)

        # --- proximidad (por pares) ---
        dist_min = float("inf")
        cerca = False
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a, b = tracks[i], tracks[j]
                d = math.dist(a.centro, b.centro)
                ancho_medio = max((a.ancho + b.ancho) / 2.0, 1.0)
                dist_min = min(dist_min, d / ancho_medio)
                if d < self.factor_prox * ancho_medio:
                    cerca = True

        self._marcar("proximidad", cerca, t)
        self._marcar("agitacion", vel_max > self._umbral_velocidad(), t)
        self._marcar("aspecto", aspecto_bajo, t)
        self.valores = {"vel_max": vel_max, "dist_min": dist_min, "aspecto_min": aspecto_min}

        # limpiar tracks que ya no estan
        vivos = {tk.canon for tk in tracks}
        for canon in list(self.previo):
            if canon not in vivos and t - self.previo[canon][0] > 5.0:
                self.previo.pop(canon, None)
                self.aspectos.pop(canon, None)

    # ------------------------------------------------------------ consulta --
    def senales(self, t: float) -> dict:
        """
        Dos umbrales distintos a proposito:

          activa (>=1 senal)  -> ESCALAR a S2. Encender el modelo de violencia
                                 cuesta CPU, no una alerta: conviene ser permisivo.
          evento (>=2 senales)-> EMITIR una alerta. Aqui hay que ser estricto.

        La proximidad SOLA no basta para alertar: en una bodega el cliente esta
        siempre cerca del que atiende, es el patron normal. Combinada si:
          proximidad + agitacion -> forcejeo
          proximidad + aspecto   -> sometimiento (agachado, manos arriba)
          agitacion  + aspecto   -> movimiento violento con cambio de postura
        """
        prox = self._sostenida("proximidad", t, self.prox_min)
        agit = self._sostenida("agitacion", t, self.sostenido_min)
        asp = self._sostenida("aspecto", t, self.sostenido_min)
        n = int(prox) + int(agit) + int(asp)
        return {
            "proximidad": prox,
            "agitacion": agit,
            "aspecto": asp,
            "n_activas": n,
            "activa": n >= 1,
            "evento": n >= 2,
            "vel_max": self.valores["vel_max"],
            "dist_min": self.valores["dist_min"],
            "umbral_vel": self._umbral_velocidad(),
        }

    def reiniciar(self) -> None:
        self.previo.clear()
        self.aspectos.clear()
        for k in self.desde:
            self.desde[k] = None
