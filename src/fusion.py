"""
Maquina de estados + reglas de fusion -> evento.

Compuertas (el costo de computo depende enteramente de esto):

  S0 VIGILIA    people_count <= 1   : personas @2 fps sin tracker + latido de armas
  S1 ACTIVIDAD  people_count >= 2   : + tracker @5 fps + armas sobre recortes
  S2 EVALUACION arma o cinematica   : + MC3-18

"Siempre hay una persona atendiendo" se usa como CONSTANTE:
    atacantes = people_count - atendientes_fijos
No hace falta un modelo espacial aprendido para restar uno. Se evaluo
auto-calibrar la zona de atencion con un mapa de calor y se descarto: activa en
el mismo instante (people_count >= 2) y solo resolvia un escenario -- que el que
atiende se haya ido al fondo -- que el latido de armas cubre mas barato.

Las reglas salen de RESUMEN seccion 4.1 y de la tabla del informe de personas.
Una senal aislada nunca dispara: todo exige persistencia temporal.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .evento import PRIORIDAD, Evento

# La cinematica es una senal booleana, asi que no tiene una probabilidad propia.
# Este valor nominal rellena `confidence`; debe calibrarse en Fase 3 con video
# propio, comparandolo contra lo que reporte MC3-18 en las mismas ventanas.
CONF_CINEMATICA = 0.70
# Igual de nominal: "apuntar + manos arriba" es booleano. Se le da mas peso que
# a la cinematica porque la combinacion es mucho mas especifica, pero el numero
# debe calibrarse en Fase 3 con video propio.
CONF_POSE = 0.80


def _punto_en_poligono(punto, poligono) -> bool:
    x, y = punto
    dentro = False
    n = len(poligono)
    for i in range(n):
        x1, y1 = poligono[i]
        x2, y2 = poligono[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            dentro = not dentro
    return dentro


def _parsear_zonas(crudo: dict) -> dict[str, list[tuple[float, float]]]:
    zonas = {}
    for nombre, texto in (crudo or {}).items():
        pts = []
        for par in str(texto).split():
            x, y = par.split(",")
            pts.append((float(x), float(y)))
        if len(pts) >= 3:
            zonas[nombre] = pts
    return zonas


class Cerebro:
    def __init__(self, cfg):
        self.cfg = cfg
        f = cfg.fusion
        self.ventana = float(f.ventana_decision_seg)
        self.cada = float(f.evaluar_cada_seg)
        self.atendientes = int(f.atendientes_fijos)
        self.cooldown = float(f.cooldown_seg)
        self.peso_arma = float(f.peso_confianza.arma)
        self.peso_viol = float(f.peso_confianza.violencia)
        self.umbral_fight = float(cfg.violencia.umbral_fight)
        self.histeresis = float(cfg.personas.histeresis_salida_seg)
        self.zonas = _parsear_zonas(cfg.get("zonas"))

        vig = cfg.personas.vigilia
        self.hist_vigilia: deque[bool] = deque(maxlen=int(vig.muestras_ventana))
        self.confirmacion = int(vig.muestras_confirmacion)

        self.estado = "S0"
        self.t_ultimo_multi = float("-inf")
        self.t_ultima_senal = float("-inf")
        self.transiciones: list[tuple[float, str, str]] = []

        self.hist_viol: deque[tuple[float, float]] = deque(maxlen=120)
        self.hist_cinem: deque[tuple[float, bool]] = deque(maxlen=600)
        self.hist_amenaza: deque[tuple[float, bool]] = deque(maxlen=600)
        self.hist_pc: deque[tuple[float, int]] = deque(maxlen=600)
        self.pose_como_atraco = bool(cfg.get("pose", {}).get("amenaza_como_atraco", True))

        # --- umbral RELATIVO de violencia ---------------------------------
        # Sobre los videos de prueba el modelo dio p50 = 0.977 en 49 ventanas:
        # un umbral absoluto es inutil contra un modelo tan sesgado al alza.
        # Pero SI discrimina en relativo (Video2 fue de 0.234 en calma a 0.977
        # en la agresion). Asi que se aprende la linea base de ESTA camara y se
        # alerta por desviacion. La linea base solo se alimenta en S1, que por
        # definicion es "hay gente pero ninguna senal de riesgo".
        ur = cfg.violencia.get("umbral_relativo", {})
        self.rel_activo = bool(ur.get("activo", False))
        self.rel_percentil = float(ur.get("percentil", 90))
        self.rel_margen = float(ur.get("margen", 0.04))
        self.rel_min = int(ur.get("muestras_min", 40))
        self.base_viol: deque[float] = deque(maxlen=int(ur.get("ventana", 600)))

        self.ultimo_evento: tuple[str, float] | None = None

    # ------------------------------------------------------------ estados --
    def _ir_a(self, nuevo: str, t: float) -> None:
        if nuevo != self.estado:
            self.transiciones.append((t, self.estado, nuevo))
            self.estado = nuevo

    def observar_vigilia(self, t: float, conteo_crudo: int) -> None:
        """S0: conteo simple de cajas. 2 de 3 muestras con >=2 personas -> S1."""
        self.hist_vigilia.append(conteo_crudo >= 2)
        if sum(self.hist_vigilia) >= self.confirmacion:
            self.hist_vigilia.clear()
            self.t_ultimo_multi = t
            self._ir_a("S1", t)

    def observar_actividad(self, t: float, people_count: int) -> None:
        """S1/S2: se sale a S0 solo tras la histeresis, para no ir y volver."""
        if people_count >= 2:
            self.t_ultimo_multi = t
        elif t - self.t_ultimo_multi > self.histeresis:
            self._ir_a("S0", t)

    def observar_senales(self, t: float, arma_presente: bool, cinem: dict, pose: dict) -> None:
        # Al historial que alimenta las REGLAS solo entra la version estricta
        # (>=2 senales); la version laxa solo sirve para escalar de estado.
        n_pose = int(pose.get("n_activas", 0))
        self.hist_cinem.append((t, bool(cinem.get("evento")) or n_pose >= 2))
        self.hist_amenaza.append((t, bool(pose.get("amenaza")), bool(pose.get("apuntar"))))
        cinem_activa = bool(cinem.get("activa")) or n_pose >= 1
        if arma_presente or cinem_activa:
            self.t_ultima_senal = t
            if self.estado == "S1":
                self._ir_a("S2", t)
        elif self.estado == "S2" and t - self.t_ultima_senal > self.histeresis:
            self._ir_a("S1", t)

    def registrar_violencia(self, t: float, p_fight: float) -> None:
        self.hist_viol.append((t, float(p_fight)))
        if self.estado == "S1":          # solo aprende de tramos sin senal de riesgo
            self.base_viol.append(float(p_fight))

    def umbral_violencia(self) -> float:
        """Absoluto mientras no haya linea base suficiente; relativo despues."""
        if not self.rel_activo or len(self.base_viol) < self.rel_min:
            return self.umbral_fight
        base = float(np.percentile(self.base_viol, self.rel_percentil))
        # se acota para que una linea base rara no deje el sistema ni sordo ni histerico
        return float(min(0.995, max(self.umbral_fight * 0.5, base + self.rel_margen)))

    # ------------------------------------------------------------- reglas --
    def evaluar(
        self,
        t: float,
        people_count: int,
        arma_sostenida: bool,
        arma_conf: float,
        caja_grupo=None,
    ) -> Evento | None:
        t0 = t - self.ventana

        # El conteo que manda es el MAXIMO de la ventana, no el instantaneo:
        # durante un forcejeo las personas se ocluyen entre si y el conteo cae
        # momentaneamente. Usar el instantaneo degradaria la clasificacion
        # justo en el momento del evento.
        self.hist_pc.append((t, int(people_count)))
        people_count = max(pc for ts, pc in self.hist_pc if ts >= t0)

        p_viol = max((p for ts, p in self.hist_viol if ts >= t0), default=0.0)
        viol_modelo = p_viol >= self.umbral_violencia()
        cinem = any(v for ts, v in self.hist_cinem if ts >= t0)
        violencia = viol_modelo or cinem
        amenaza = (self.pose_como_atraco
                   and any(a for ts, a, _ in self.hist_amenaza if ts >= t0))
        apuntando = any(ap for ts, _, ap in self.hist_amenaza if ts >= t0)

        # La postura de apuntar vale como evidencia de arma: el arma mide 20-40 px
        # y se pierde en el 71-86% de los cuadros; el gesto mide cientos de px.
        evidencia_arma = arma_sostenida or amenaza or apuntando
        atacantes = max(0, people_count - self.atendientes)

        if evidencia_arma and violencia and atacantes >= 2:
            tipo, sev = "possible_multiple_robbery", "high"
        elif evidencia_arma and violencia:
            tipo, sev = "possible_robbery", "high"
        elif amenaza:
            # uno apuntando y otro con las manos arriba, sostenido: no hace
            # falta nada mas para llamarlo posible atraco
            tipo, sev = "possible_robbery", "high"
        elif arma_sostenida:
            tipo, sev = "suspicious_weapon", "medium"
        elif violencia:
            tipo, sev = "physical_violence", "medium"
        else:
            return None

        if not self._pasa_cooldown(tipo, t):
            return None

        conf_viol = max(p_viol if viol_modelo else 0.0, CONF_CINEMATICA if cinem else 0.0)
        conf_arma_ef = max(arma_conf if arma_sostenida else 0.0,
                           CONF_POSE if amenaza else 0.0,
                           CONF_POSE * 0.8 if apuntando else 0.0)
        if tipo in ("possible_robbery", "possible_multiple_robbery"):
            confianza = (self.peso_arma * conf_arma_ef + self.peso_viol * conf_viol
                         if violencia else conf_arma_ef)
        elif tipo == "suspicious_weapon":
            confianza = arma_conf
        else:
            confianza = conf_viol

        detecciones = []
        if arma_sostenida:
            detecciones.append({"type": "weapon", "confidence": round(arma_conf, 4)})
        if amenaza:
            detecciones.append({"type": "aiming_posture", "confidence": CONF_POSE})
        elif apuntando:
            detecciones.append({"type": "aiming_posture", "confidence": round(CONF_POSE * 0.8, 4)})
        if viol_modelo:
            detecciones.append({"type": "physical_violence", "confidence": round(p_viol, 4)})
        if cinem:
            detecciones.append({"type": "kinematic_struggle", "confidence": CONF_CINEMATICA})

        self.ultimo_evento = (tipo, t)
        return Evento(
            event_type=tipo,
            severity=sev,
            confidence=confianza,
            detected_at=Evento.ahora_iso(),
            t_video=t,
            people_count=people_count,
            atacantes=atacantes,
            detections=detecciones,
            zone=self._zona(caja_grupo),
            business_id=self.cfg.identidad.business_id,
            camera_id=self.cfg.identidad.camera_id,
        )

    def _pasa_cooldown(self, tipo: str, t: float) -> bool:
        if self.ultimo_evento is None:
            return True
        tipo_ant, t_ant = self.ultimo_evento
        if t - t_ant > self.cooldown:
            return True
        # dentro del cooldown solo pasa una escalada de gravedad
        return PRIORIDAD.get(tipo, 0) > PRIORIDAD.get(tipo_ant, 0)

    def _zona(self, caja_grupo) -> str | None:
        if not self.zonas or caja_grupo is None:
            return None
        x1, y1, x2, y2 = caja_grupo
        centro = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        for nombre, poli in self.zonas.items():
            if _punto_en_poligono(centro, poli):
                return nombre
        return None
