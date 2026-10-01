"""
Clasificacion de violencia fisica (MC3-18, clases fight / noFight).

Tres correcciones respecto al codigo del repositorio original:

1. weights=None al construir la arquitectura. El repo usa pretrained=True, que
   descarga ~50 MB de pesos Kinetics en CADA arranque solo para sobreescribirlos
   con el state_dict local. Rompe el arranque sin internet en la tienda.

2. Se devuelve la PROBABILIDAD. El original usa PredTopKClass(), que descarta el
   softmax; por eso el pipeline nunca emitio un valor de confianza, pese a que
   RESUMEN seccion 13 lo exige. (Los numeros 0.9966 / 0.3833 que aparecen en el
   informe de validacion no son confianzas: son el print(end-start) de infer.py,
   es decir segundos de ejecucion.)

3. Muestreo temporal correcto. El entrenamiento toma 16 cuadros repartidos sobre
   TODO el clip (~4.8 s, ~3.3 fps efectivos); el infer.py del repo usa skip=2,
   que da una ventana de 1.07 s. El movimiento se ve ~4.5x mas rapido de lo que
   la red aprendio. Aqui la ventana la fija fuente_video.muestrear().

albumentations se reemplaza por cv2 + numpy: solo hacian falta resize,
center-crop y normalize.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torchvision

# Misma transformacion que A.Resize(128,171) + A.CenterCrop(112,112) + A.Normalize
ALTO_R, ANCHO_R = 128, 171
LADO = 112
Y0, X0 = (ALTO_R - LADO) // 2, (ANCHO_R - LADO) // 2
MEDIA = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
DESV = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

# Fraccion del cuadro que sobrevive al center-crop
FRAC_X, FRAC_Y = LADO / ANCHO_R, LADO / ALTO_R


def _preparar(cuadro_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(cuadro_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (ANCHO_R, ALTO_R), interpolation=cv2.INTER_LINEAR)
    img = img[Y0 : Y0 + LADO, X0 : X0 + LADO]
    return (img.astype(np.float32) / 255.0 - MEDIA) / DESV


def _recorte_encuadrado(cuadro, caja) -> np.ndarray:
    """
    Expande la caja del grupo de personas de modo que, tras resize+center-crop,
    quede exactamente dentro de la vista de 112x112 que ve el modelo.
    Sin esto el center-crop recortaria el grupo.
    """
    alto, ancho = cuadro.shape[:2]
    x1, y1, x2, y2 = caja
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w = max(16.0, (x2 - x1)) / FRAC_X
    h = max(16.0, (y2 - y1)) / FRAC_Y
    rx1 = max(0, int(cx - w / 2))
    ry1 = max(0, int(cy - h / 2))
    rx2 = min(ancho, int(cx + w / 2))
    ry2 = min(alto, int(cy + h / 2))
    if rx2 - rx1 < 16 or ry2 - ry1 < 16:
        return cuadro
    return cuadro[ry1:ry2, rx1:rx2]


class DetectorViolencia:
    def __init__(self, cfg):
        self.cfg = cfg.violencia
        self.umbral = float(self.cfg.umbral_fight)
        self.n = int(self.cfg.frames_secuencia)
        self.paso_seg = int(self.cfg.stride_sobre_base) / float(cfg.fuente.fps_base)
        self.recorte = str(self.cfg.recorte)
        self.comparar = bool(self.cfg.comparar_ambos)

        modelo = torchvision.models.video.mc3_18(weights=None)
        modelo.fc = torch.nn.Linear(modelo.fc.in_features, 2)
        estado = torch.load(str(cfg.ruta(self.cfg.modelo)), map_location="cpu", weights_only=True)
        modelo.load_state_dict(estado)
        modelo.eval()
        self.modelo = modelo

        # CLASSES_LIST del repo original: indice 0 = fight
        self.idx_fight = 0
        self.ultimo: dict[str, float | None] = {"centro": None, "personas": None}

    def ventana_segundos(self) -> float:
        return (self.n - 1) * self.paso_seg

    def region_vista(self, forma, caja_grupo=None) -> tuple[int, int, int, int]:
        """
        Que parte del cuadro MIRA realmente el modelo, en coordenadas de ese
        cuadro. No es una deteccion -- es el area de analisis, y conviene
        dibujarla porque el center-crop deja fuera la periferia de la escena.
        """
        alto, ancho = forma[:2]
        if self.recorte == "personas" and caja_grupo is not None:
            return tuple(int(v) for v in caja_grupo)
        cw, ch = ancho * FRAC_X, alto * FRAC_Y
        return (
            int((ancho - cw) / 2), int((alto - ch) / 2),
            int((ancho + cw) / 2), int((alto + ch) / 2),
        )

    def _tensor(self, cuadros: list[np.ndarray]) -> torch.Tensor:
        pila = np.stack([_preparar(c) for c in cuadros])        # (T,112,112,3)
        pila = np.transpose(pila, (3, 0, 1, 2))                  # (3,T,112,112)
        return torch.from_numpy(np.ascontiguousarray(pila))

    def evaluar(self, cuadros: list[np.ndarray], caja_grupo=None) -> dict[str, float | None]:
        """
        Devuelve {'centro': p_fight, 'personas': p_fight | None}.
        `caja_grupo` va en coordenadas de los cuadros recibidos.
        """
        variantes: list[str] = []
        tensores: list[torch.Tensor] = []

        quiere_personas = (self.recorte == "personas" or self.comparar) and caja_grupo is not None
        quiere_centro = self.recorte == "centro" or self.comparar or not quiere_personas

        if quiere_centro:
            variantes.append("centro")
            tensores.append(self._tensor(cuadros))
        if quiere_personas:
            variantes.append("personas")
            tensores.append(self._tensor([_recorte_encuadrado(c, caja_grupo) for c in cuadros]))

        with torch.no_grad():
            lote = torch.stack(tensores)                         # (B,3,T,112,112)
            probs = torch.softmax(self.modelo(lote), dim=1)[:, self.idx_fight]

        salida: dict[str, float | None] = {"centro": None, "personas": None}
        for nombre, p in zip(variantes, probs.tolist()):
            salida[nombre] = float(p)
        self.ultimo = salida
        return salida

    def p_decision(self, resultado: dict[str, float | None]) -> float:
        """La probabilidad que manda en las reglas (la otra solo se registra)."""
        p = resultado.get(self.recorte)
        if p is None:
            p = resultado.get("centro") or resultado.get("personas") or 0.0
        return float(p)
