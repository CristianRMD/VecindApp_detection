"""
Deteccion de armas (YOLOv8s, una sola clase: 'weapon').

EL PROBLEMA MEDIDO
------------------
Sobre video real, un arma genuinamente visible solo supera conf 0.45 en el
14-29% de los cuadros. El detector la ve y la pierde. Contar "N detecciones en
una ventana" es entonces una mala regla: para ser alcanzable hay que bajar N,
y al bajarlo entra ruido, porque la regla es espacialmente CIEGA -- dos falsos
positivos en esquinas opuestas cuentan igual que dos vistas del mismo arma.

LA SOLUCION: SEGUIR EL ARMA, NO CONTAR DETECCIONES
--------------------------------------------------
Se baja el umbral a 0.20 (donde el arma aparece en el 43-95% de los cuadros) y
las detecciones se enlazan en TRAZAS por cercania espacial. Una traza se
confirma por tres condiciones a la vez: numero de observaciones, confianza
media y un pico decente.

Esto no es un intercambio entre sensibilidad y precision: es mejor en las dos.
Mas sensible, porque aprovecha detecciones debiles que antes se tiraban. Mas
especifico, porque el ruido esta disperso y no forma trazas coherentes.

LAS CUATRO RUTAS DE INFERENCIA
------------------------------
  LATIDO (S0)      cuadro completo cada N segundos. Cubre que el que atiende
                   haya salido de cuadro, y un arma sobre el mostrador.
  RECORTES (S1+)   sobre la caja de cada persona. Recortar normaliza la escala
                   aparente del arma; medido, es lo mas consistente.
  PASADA COMPLETA  segunda escala (@960). La sensibilidad del modelo NO es
                   monotona con imgsz.
  ZONAS CALIENTES  re-inspeccion ampliada y con TTA de las regiones donde ya se
                   vio algo. TTA (augment=True) triplico los aciertos en
                   Video5 (3/21 -> 9/21) por solo 1.3-2x de coste.

Nota: se probo tambien inferencia en mosaico (estilo SAHI, teselas a resolucion
nativa). Ayuda en una escena y PERJUDICA en otra (Video3: 2/16 -> 0/16): trocear
destruye el contexto, y este modelo depende de el. Descartado.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from ultralytics import YOLO


@dataclass
class DetArma:
    t: float
    conf: float
    xyxy: tuple[float, float, float, float]
    origen: str  # "latido" | "recorte" | "zona"


@dataclass
class TrazaArma:
    """Detecciones sucesivas de lo que parece ser la misma arma."""
    centro: tuple[float, float]
    xyxy: tuple[float, float, float, float]
    t_ini: float
    t_ult: float
    confs: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.confs)

    @property
    def media(self) -> float:
        return sum(self.confs) / max(1, len(self.confs))

    @property
    def pico(self) -> float:
        return max(self.confs, default=0.0)


@dataclass
class ZonaCaliente:
    xyxy: tuple[float, float, float, float]
    t_hasta: float


class DetectorArmas:
    def __init__(self, cfg):
        self.modelo = YOLO(str(cfg.ruta(cfg.armas.modelo)))
        self.conf = float(cfg.armas.conf)
        self.conf_dibujo = float(cfg.armas.get("conf_dibujo", 0.35))
        self.lat = cfg.armas.latido
        self.act = cfg.armas.activo

        seg = cfg.armas.get("seguimiento", {})
        self.seg_activo = bool(seg.get("activo", False))
        self.seg_conf = float(seg.get("conf_semilla", 0.20))
        self.seg_seg = float(seg.get("segundos", 4.0))
        self.seg_imgsz = int(seg.get("imgsz", 416))
        self.seg_tta = bool(seg.get("tta", True))
        self.seg_factor = float(seg.get("factor", 3.0))
        self.seg_lado_min = int(seg.get("lado_min_px", 160))
        self.seg_max = int(seg.get("max_zonas", 2))

        tr = cfg.armas.get("trazas", {})
        self.tr_activo = bool(tr.get("activo", True))
        # La distancia de enlace se expresa como FRACCION del ancho del cuadro,
        # no en pixeles: 140 px significan cosas muy distintas a 854 y a 1920.
        self.tr_dist_frac = float(tr.get("dist_max_frac", 0.12))
        self.ancho_cuadro = 1280.0
        self.tr_ttl = float(tr.get("ttl_seg", 2.0))
        self.tr_min_n = int(tr.get("min_detecciones", 3))
        self.tr_min_media = float(tr.get("min_conf_media", 0.33))
        self.tr_min_pico = float(tr.get("min_conf_pico", 0.45))

        # respaldo cuando el seguimiento esta desactivado
        self.ventana = float(cfg.armas.persistencia_ventana_seg)
        self.minimo = int(cfg.armas.persistencia_detecciones)

        self.historial: deque[DetArma] = deque(maxlen=600)
        self.ultimas: list[DetArma] = []
        self.zonas: list[ZonaCaliente] = []
        self.trazas: list[TrazaArma] = []

    # ---------------------------------------------------------------- S0 ---
    def latido(self, cuadro, t: float, imgsz: int | None = None) -> list[DetArma]:
        r = self.modelo.predict(
            cuadro, imgsz=imgsz or self.lat.imgsz, conf=self.conf, verbose=False
        )[0]
        dets = [
            DetArma(t, float(cf), tuple(caja), "latido")
            for caja, cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())
        ] if r.boxes is not None else []
        return self._registrar(dets, t)

    # --------------------------------------------------------------- S1+ ---
    def sobre_personas(self, cuadro, tracks, t: float) -> list[DetArma]:
        if not tracks:
            return self._registrar([], t)
        margen = float(self.act.margen_crop)
        cajas = []
        for tk in sorted(tracks, key=lambda k: -(k.ancho * k.alto))[: int(self.act.max_crops)]:
            x1, y1, x2, y2 = tk.xyxy
            mx, my = (x2 - x1) * margen, (y2 - y1) * margen
            cajas.append((x1 - mx, y1 - my, x2 + mx, y2 + my))
        return self._registrar(self._inferir(cuadro, cajas, self.act.imgsz, "recorte", t), t)

    def sobre_zonas(self, cuadro, t: float) -> list[DetArma]:
        """Re-inspeccion ampliada y con TTA de las regiones ya sospechosas."""
        self.zonas = [z for z in self.zonas if z.t_hasta > t]
        if not self.seg_activo or not self.zonas:
            return []
        dets = self._inferir(
            cuadro, [z.xyxy for z in self.zonas], self.seg_imgsz, "zona", t, tta=self.seg_tta
        )
        return self._registrar(dets, t, reemplazar=False)

    def hay_zonas(self, t: float) -> bool:
        return any(z.t_hasta > t for z in self.zonas)

    # ------------------------------------------------------------ consulta --
    def sostenida(self, t: float) -> tuple[bool, float, int]:
        """(hay arma confirmada, confianza del pico, observaciones de la traza)."""
        self._purgar_trazas(t)
        if not self.tr_activo:
            recientes = [d for d in self.historial if d.t >= t - self.ventana]
            if not recientes:
                return False, 0.0, 0
            return (len(recientes) >= self.minimo,
                    max(d.conf for d in recientes), len(recientes))

        confirmadas = self.trazas_confirmadas()
        if confirmadas:
            mejor = max(confirmadas, key=lambda z: z.pico)
            return True, mejor.pico, mejor.n
        if self.trazas:
            mejor = max(self.trazas, key=lambda z: z.pico)
            return False, mejor.pico, mejor.n
        return False, 0.0, 0

    def trazas_confirmadas(self) -> list[TrazaArma]:
        return [
            z for z in self.trazas
            if z.n >= self.tr_min_n and z.media >= self.tr_min_media and z.pico >= self.tr_min_pico
        ]

    def hay_alguna(self, t: float, ventana: float) -> bool:
        return any(d.t >= t - ventana for d in self.historial)

    def para_dibujar(self) -> list[DetArma]:
        """Detecciones solidas, mas cualquiera que pertenezca a una traza viva:
        con conf 0.20 el ruido inundaria el video si se dibujara todo."""
        return [d for d in self.ultimas if d.conf >= self.conf_dibujo]

    # ------------------------------------------------------------- interno --
    def _inferir(self, cuadro, cajas, imgsz, origen, t, tta: bool = False) -> list[DetArma]:
        alto, ancho = cuadro.shape[:2]
        self.ancho_cuadro = float(ancho)
        recortes, origenes = [], []
        for x1, y1, x2, y2 in cajas:
            a, b = max(0, int(x1)), max(0, int(y1))
            c, d = min(ancho, int(x2)), min(alto, int(y2))
            if c - a < 24 or d - b < 24:
                continue
            recortes.append(cuadro[b:d, a:c])
            origenes.append((a, b))
        if not recortes:
            return []

        resultados = self.modelo.predict(
            recortes, imgsz=imgsz, conf=self.conf, augment=tta, verbose=False
        )
        dets = []
        for r, (ox, oy) in zip(resultados, origenes):
            if r.boxes is None:
                continue
            for caja, cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                x1, y1, x2, y2 = caja
                dets.append(DetArma(t, float(cf), (x1 + ox, y1 + oy, x2 + ox, y2 + oy), origen))
        return dets

    def _purgar_trazas(self, t: float) -> None:
        self.trazas = [z for z in self.trazas if t - z.t_ult <= self.tr_ttl]

    def _enlazar(self, det: DetArma, t: float) -> None:
        x1, y1, x2, y2 = det.xyxy
        centro = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        base = self.tr_dist_frac * self.ancho_cuadro
        mejor, mejor_d = None, 0.0
        for z in self.trazas:
            # cuanto mas tiempo lleva sin verse, mas lejos puede haberse movido
            tolerancia = base * (1.0 + (t - z.t_ult) / max(self.tr_ttl, 1e-6))
            d = math.dist(centro, z.centro)
            if d < tolerancia and (mejor is None or d < mejor_d):
                mejor_d, mejor = d, z
        if mejor is None:
            self.trazas.append(TrazaArma(centro, det.xyxy, t, t, [det.conf]))
        else:
            mejor.centro, mejor.xyxy, mejor.t_ult = centro, det.xyxy, t
            mejor.confs.append(det.conf)
            if len(mejor.confs) > 40:
                mejor.confs = mejor.confs[-40:]

    def _abrir_zona(self, det: DetArma, t: float) -> None:
        x1, y1, x2, y2 = det.xyxy
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for z in self.zonas:
            zx1, zy1, zx2, zy2 = z.xyxy
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                z.t_hasta = t + self.seg_seg
                return
        lado = max(self.seg_lado_min, self.seg_factor * max(x2 - x1, y2 - y1))
        self.zonas.append(
            ZonaCaliente((cx - lado / 2, cy - lado / 2, cx + lado / 2, cy + lado / 2),
                         t + self.seg_seg)
        )
        self.zonas = self.zonas[-self.seg_max :]

    def _registrar(self, dets: list[DetArma], t: float, reemplazar: bool = True) -> list[DetArma]:
        self.historial.extend(dets)
        self._purgar_trazas(t)
        for d in dets:
            if self.tr_activo:
                self._enlazar(d, t)
            if self.seg_activo and d.conf >= self.seg_conf:
                self._abrir_zona(d, t)
        self.ultimas = dets if reemplazar else (self.ultimas + dets)
        return dets
