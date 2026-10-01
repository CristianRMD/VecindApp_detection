"""
Deteccion y conteo de personas.

Dos modos segun el estado:

  S0 VIGILIA  -> contar()  : solo cuenta cajas, SIN tracker.
                 En vigilia no hay nada que seguir: la pregunta no es
                 "quien es quien" sino "hay mas de una persona". Contar
                 funciona a 2 fps; el tracker no (a esa tasa la asociacion
                 por IoU de ByteTrack se rompe).

  S1+ ACTIVO  -> seguir()  : YOLOv8n + ByteTrack + las reglas de conteo
                 validadas en el ticket de personas: ventana temporal,
                 persistencia minima y deduplicacion de tracks que
                 reaparecen tras una oclusion corta.

people_count NUNCA es el conteo crudo de tracks activos en un cuadro: el
detector genera ruido de corta duracion que no corresponde a personas reales.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Track:
    tid: int                              # id crudo de ByteTrack
    canon: int                            # id canonico tras deduplicacion
    xyxy: tuple[float, float, float, float]
    centro: tuple[float, float]
    t: float
    kpts: object = None                   # (17,3) de YOLOv8-pose, o None

    @property
    def ancho(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def alto(self) -> float:
        return self.xyxy[3] - self.xyxy[1]


class DetectorPersonas:
    def __init__(self, cfg):
        ruta = str(cfg.ruta(cfg.personas.modelo))
        # Dos instancias a proposito: mezclar predict() y track() sobre el mismo
        # objeto deja activos los callbacks del tracker durante el predict.
        self.m_conteo = YOLO(ruta)

        # En S1+ se usa el modelo de POSE: da las mismas cajas MAS 17 puntos
        # clave por persona, por solo un 9% mas de coste (23.3 vs 21.4 ms).
        # No se suma al detector normal: lo reemplaza.
        ruta_pose = cfg.personas.get("modelo_pose")
        self.con_pose = bool(cfg.personas.get("usar_pose", False)) and bool(ruta_pose)
        self.m_track = YOLO(str(cfg.ruta(ruta_pose)) if self.con_pose else ruta)

        self.clase = cfg.personas.clase_persona
        self.vig = cfg.personas.vigilia
        self.act = cfg.personas.activo

        self.ventana = float(self.act.ventana_seg)
        self.persistencia = float(self.act.persistencia_min)
        self.dedupe_gap = float(self.act.dedupe_gap_seg)
        self.dedupe_dist = float(self.act.dedupe_dist_px)

        # observaciones: (t, conjunto de ids canonicos vistos en ese muestreo)
        self.obs: deque[tuple[float, frozenset]] = deque(maxlen=400)
        self.canon: dict[int, int] = {}
        self.ultimo_visto: dict[int, float] = {}
        self.ultimo_centro: dict[int, tuple[float, float]] = {}
        self.terminados: list[tuple[int, float, tuple[float, float]]] = []

        self.tracks_actuales: list[Track] = []
        self.id_switches = 0

    # --------------------------------------------------------------- S0 ----
    def contar(self, cuadro) -> int:
        r = self.m_conteo.predict(
            cuadro,
            imgsz=self.vig.imgsz,
            conf=self.vig.conf,
            classes=[self.clase],
            verbose=False,
        )[0]
        return 0 if r.boxes is None else len(r.boxes)

    # --------------------------------------------------------------- S1+ ---
    def seguir(self, cuadro, t: float) -> list[Track]:
        r = self.m_track.track(
            cuadro,
            imgsz=self.act.imgsz,
            conf=self.act.conf,
            classes=[self.clase],
            persist=True,
            tracker=self.act.tracker,
            verbose=False,
        )[0]

        tracks: list[Track] = []
        vistos_ahora: set[int] = set()

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            cajas = r.boxes.xyxy.tolist()
            kps = None
            if self.con_pose and getattr(r, "keypoints", None) is not None:
                kps = r.keypoints.data.cpu().numpy()   # (N, 17, 3), mismo orden que las cajas
            for i, (tid, caja) in enumerate(zip(ids, cajas)):
                x1, y1, x2, y2 = caja
                centro = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                canon = self._canonico(tid, t, centro)
                vistos_ahora.add(tid)
                self.ultimo_visto[tid] = t
                self.ultimo_centro[tid] = centro
                kp = kps[i] if kps is not None and i < len(kps) else None
                tracks.append(Track(tid, canon, (x1, y1, x2, y2), centro, t, kp))

        self._retirar(t, vistos_ahora)
        self.obs.append((t, frozenset(tk.canon for tk in tracks)))
        self.tracks_actuales = tracks
        return tracks

    def people_count(self, t: float) -> int:
        """Conteo deduplicado con persistencia sobre la ventana que termina en t."""
        t0 = t - self.ventana
        en_ventana = [(ts, ids) for ts, ids in self.obs if ts >= t0]
        if not en_ventana:
            return 0
        muestras = len(en_ventana)
        minimo = max(1, round(muestras * self.persistencia))

        conteo: dict[int, int] = {}
        for _, ids in en_ventana:
            for cid in ids:
                conteo[cid] = conteo.get(cid, 0) + 1
        return sum(1 for n in conteo.values() if n >= minimo)

    def conteo_crudo(self) -> int:
        return len(self.tracks_actuales)

    def reiniciar_tracker(self) -> None:
        """Al entrar en S1 el estado del tracker puede venir viciado."""
        try:
            for tr in self.m_track.predictor.trackers:
                tr.reset()
        except Exception:
            pass
        self.obs.clear()
        self.canon.clear()
        self.ultimo_visto.clear()
        self.ultimo_centro.clear()
        self.terminados.clear()
        self.tracks_actuales = []

    # ------------------------------------------------------------- interno --
    def _canonico(self, tid: int, t: float, centro: tuple[float, float]) -> int:
        if tid in self.canon:
            return self.canon[tid]

        mejor_i, mejor_d = None, self.dedupe_dist
        for i, (_, t_fin, c) in enumerate(self.terminados):
            if 0 <= t - t_fin <= self.dedupe_gap:
                d = math.dist(centro, c)
                if d < mejor_d:
                    mejor_d, mejor_i = d, i

        if mejor_i is not None:
            canon = self.terminados.pop(mejor_i)[0]
            self.id_switches += 1
        else:
            canon = tid

        self.canon[tid] = canon
        return canon

    def _retirar(self, t: float, vistos_ahora: set[int]) -> None:
        for tid in list(self.ultimo_visto):
            if tid in vistos_ahora:
                continue
            if t - self.ultimo_visto[tid] > self.dedupe_gap:
                canon = self.canon.pop(tid, tid)
                centro = self.ultimo_centro.pop(tid, (0.0, 0.0))
                self.terminados.append((canon, self.ultimo_visto.pop(tid), centro))
        # no conservar candidatos de re-vinculacion mas viejos que la ventana
        self.terminados = [e for e in self.terminados if t - e[1] <= self.dedupe_gap * 2]

    # --------------------------------------------------------------- util --
    def caja_grupo(self, margen: float = 0.15) -> tuple[int, int, int, int] | None:
        """Union de las cajas de persona actuales, con margen. Para el recorte
        de violencia y para atribuir espacialmente el evento."""
        if not self.tracks_actuales:
            return None
        xs1 = [tk.xyxy[0] for tk in self.tracks_actuales]
        ys1 = [tk.xyxy[1] for tk in self.tracks_actuales]
        xs2 = [tk.xyxy[2] for tk in self.tracks_actuales]
        ys2 = [tk.xyxy[3] for tk in self.tracks_actuales]
        x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
        mx, my = (x2 - x1) * margen, (y2 - y1) * margen
        return int(x1 - mx), int(y1 - my), int(x2 + mx), int(y2 + my)
