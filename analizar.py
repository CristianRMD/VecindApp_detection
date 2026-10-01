"""
Agente de deteccion - analisis de video.

Uso:
    python analizar.py                              # todos los .mp4 de videos/entrada/
    python analizar.py --video videos/entrada/x.mp4
    python analizar.py --video x.mp4 --inicio 10 --fin 40
    python analizar.py --sin-video                  # sin video anotado (mas rapido)

Salidas por cada video:
    videos/salida/<nombre>_anotado.mp4    cuadros con cajas, estado y senales
    videos/salida/<nombre>_ventanas.csv   una fila por evaluacion de fusion
    eventos/<event_id>.json               evento segun RESUMEN seccion 13
    eventos/<event_id>.mp4                clip de evidencia
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path

import cv2

RAIZ = Path(__file__).parent
sys.path.insert(0, str(RAIZ))

from src import config as cfg_mod                     # noqa: E402
from src import hud                                   # noqa: E402
from src.cinematica import Cinematica                 # noqa: E402
from src.det_armas import DetectorArmas               # noqa: E402
from src.det_personas import DetectorPersonas         # noqa: E402
from src.det_violencia import DetectorViolencia       # noqa: E402
from src.evidencia import Evidencia                   # noqa: E402
from src.fuente_video import FuenteVideo              # noqa: E402
from src.fusion import Cerebro                        # noqa: E402
from src.pose import ESQUELETO, AnalizadorPose        # noqa: E402

EXTENSIONES = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg")

VERDE = (80, 220, 80)        # personas
ROJO = (60, 60, 255)         # armas
NARANJA = (0, 170, 255)      # zona de confirmacion de arma
CIAN = (230, 200, 60)        # zona que analiza el modelo de violencia
HUESO = (170, 210, 170)      # esqueleto
MAGENTA = (220, 60, 255)     # gesto de apuntar
AMARILLO = (0, 230, 230)     # manos arriba
BLANCO = (255, 255, 255)
GRIS = (190, 190, 190)

ESTADO_ES = {"S0": "VIGILANDO", "S1": "ACTIVIDAD", "S2": "EVALUANDO"}
COLOR_ESTADO = {"S0": GRIS, "S1": NARANJA, "S2": ROJO}
EVENTO_ES = {
    "suspicious_weapon": "ARMA SOSPECHOSA",
    "physical_violence": "VIOLENCIA FÍSICA",
    "possible_robbery": "POSIBLE ATRACO",
    "possible_multiple_robbery": "POSIBLE ATRACO CON VARIAS PERSONAS",
}
SENAL_ES = {"proximidad": "proximidad", "agitacion": "agitación", "aspecto": "postura"}
POSE_ES = {"apuntar": "APUNTANDO", "manos_arriba": "manos arriba", "en_suelo": "en el suelo"}
BANDA = 78          # alto del panel superior; nada de texto se dibuja por encima

COLUMNAS = [
    "t", "estado", "people_count", "atacantes",
    "arma_sostenida", "arma_conf", "arma_dets",
    "p_fight_centro", "p_fight_personas",
    "cin_proximidad", "cin_agitacion", "cin_aspecto", "cin_n", "cin_vel_max", "cin_umbral_vel",
    "pose_apuntar", "pose_manos_arriba", "pose_en_suelo", "pose_amenaza",
    "pose_n_apuntando", "pose_n_manos_arriba",
    "evento",
]


def _reloj(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def dibujar(lienzo, panel, fuente, estado, tracks, armas_ult, zonas, region_viol, ctx):
    alto, ancho = lienzo.shape[:2]
    textos = []

    # --- Zona que MIRA el modelo de violencia -------------------------------
    # No es una deteccion: es el area de analisis. Se dibuja a trazos para
    # distinguirla de las cajas solidas, que si son detecciones.
    def etiqueta(x1, y1):
        """Ninguna etiqueta invade el panel superior ni se sale por la izquierda."""
        return max(2, x1 + 2), max(BANDA + 3, y1 - 17)

    if region_viol is not None:
        x1, y1, x2, y2 = region_viol
        hud.rect_punteado(lienzo, (x1, y1), (x2, y2), CIAN, 1, 12, 8)
        # al pie de la region, para no chocar con las etiquetas de persona
        textos.append(("zona analizada (violencia)",
                       (max(2, x1 + 5), min(alto - 46, y2 - 18)), CIAN, 12, False))

    # --- Zonas de confirmacion de arma --------------------------------------
    for z in zonas:
        x1, y1, x2, y2 = fuente.a_buffer(z.xyxy)
        hud.rect_punteado(lienzo, (x1, y1), (x2, y2), NARANJA, 1, 6, 5)
        textos.append(("confirmando", etiqueta(x1, y1), NARANJA, 11, False))

    # --- Personas (deteccion con caja) --------------------------------------
    for tk in tracks:
        x1, y1, x2, y2 = fuente.a_buffer(tk.xyxy)
        cv2.rectangle(lienzo, (x1, y1), (x2, y2), VERDE, 2)
        textos.append((f"Persona {tk.canon}", etiqueta(x1, y1), VERDE, 13, False))

    # --- Esqueletos y gesto de apuntar --------------------------------------
    def punto(xy):
        x, y, _, _ = fuente.a_buffer((xy[0], xy[1], xy[0], xy[1]))
        return (x, y)

    por_canon = {p.canon: p for p in ctx["posturas"]}
    for tk in tracks:
        if tk.kpts is None:
            continue
        pts = [punto((k[0], k[1])) if float(k[2]) >= 0.5 else None for k in tk.kpts]
        for a, b in ESQUELETO:
            if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
                cv2.line(lienzo, pts[a], pts[b], HUESO, 1, cv2.LINE_AA)

        p = por_canon.get(tk.canon)
        if p is None:
            continue
        if p.brazo is not None:
            s, w = punto(p.brazo[0]), punto(p.brazo[1])
            apunta = bool(p.apunta_a)
            cv2.line(lienzo, s, w, MAGENTA if apunta else HUESO, 3, cv2.LINE_AA)
            if apunta:
                dx, dy = w[0] - s[0], w[1] - s[1]
                n = math.hypot(dx, dy) or 1.0
                fin = (int(w[0] + dx / n * 85), int(w[1] + dy / n * 85))
                cv2.arrowedLine(lienzo, w, fin, MAGENTA, 2, cv2.LINE_AA, tipLength=0.3)
                textos.append(("APUNTANDO", (max(2, w[0] - 30), max(BANDA + 3, w[1] - 22)),
                               MAGENTA, 13, True))
        if p.manos_arriba:
            bx1, by1, _, _ = fuente.a_buffer(tk.xyxy)
            textos.append(("MANOS ARRIBA", (max(2, bx1 + 2), max(BANDA + 3, by1 + 2)),
                           AMARILLO, 12, True))

    # --- Armas: trazas confirmadas (caja gruesa) y detecciones sueltas ------
    for z in ctx["trazas"]:
        x1, y1, x2, y2 = fuente.a_buffer(z.xyxy)
        cv2.rectangle(lienzo, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), ROJO, 3)
        textos.append((f"ARMA CONFIRMADA {z.pico:.0%}  ({z.n} vistas)",
                       etiqueta(x1, y1), ROJO, 13, True))

    for d in armas_ult:
        x1, y1, x2, y2 = fuente.a_buffer(d.xyxy)
        cv2.rectangle(lienzo, (x1, y1), (x2, y2), ROJO, 1)
        textos.append((f"arma {d.conf:.0%}", (max(2, x1 + 2), max(BANDA + 3, y2 + 2)),
                       ROJO, 11, False))

    # --- Panel superior ------------------------------------------------------
    hud.banda(lienzo, 0, BANDA, 0.72)
    cv2.line(lienzo, (0, BANDA), (ancho, BANDA), COLOR_ESTADO[estado], 2)

    textos.append((_reloj(ctx["t"]), (10, 4), BLANCO, 18, True))
    textos.append((ESTADO_ES[estado], (105, 4), COLOR_ESTADO[estado], 18, True))

    # pc es el conteo con persistencia (el que usan las reglas); el crudo es el
    # numero de cajas dibujadas. Se muestran los dos cuando no coinciden, para
    # que no desconcierte ver "Personas: 0" con cajas en pantalla.
    crudo = len(tracks)
    txt_pers = f"Personas: {ctx['pc']}" + (f"  ({crudo} en cuadro)" if crudo != ctx["pc"] else "")
    textos.append((txt_pers, (10, 30), BLANCO, 15, False))
    textos.append((f"Atacantes: {ctx['atacantes']}", (230, 30), BLANCO, 15, False))

    if ctx["arma_sost"]:
        txt_arma, col_arma = f"Arma: CONFIRMADA {ctx['arma_conf']:.0%}", ROJO
    elif ctx["arma_dets"]:
        txt_arma, col_arma = f"Arma: indicio {ctx['arma_conf']:.0%}", BLANCO
    else:
        txt_arma, col_arma = "Arma: sin detectar", GRIS
    textos.append((txt_arma, (10, 55), col_arma, 14, ctx["arma_sost"]))

    p = ctx["p_centro"]
    if p is None:
        txt_viol, col_viol = "Violencia: sin evaluar", GRIS
    else:
        txt_viol = f"Violencia: {p:.0%}"
        col_viol = ROJO if p >= ctx["umbral_viol"] else BLANCO
    textos.append((txt_viol, (200, 55), col_viol, 14, False))

    activas = [SENAL_ES[k] for k in ("proximidad", "agitacion", "aspecto") if ctx["cin"].get(k)]
    activas += [POSE_ES[k] for k in ("apuntar", "manos_arriba", "en_suelo")
                if ctx["pose"].get(k)]
    txt_sen = "Señales: " + (", ".join(activas) if activas else "ninguna")
    col_sen = MAGENTA if ctx["pose"].get("amenaza") else (BLANCO if activas else GRIS)
    textos.append((txt_sen, (375, 55), col_sen, 13, bool(ctx["pose"].get("amenaza"))))

    # --- Banda inferior: leyenda, o la alerta cuando la hay -----------------
    if ctx["evento"]:
        hud.banda(lienzo, alto - 38, alto, 0.80, (0, 0, 130))
        nombre = EVENTO_ES.get(ctx["evento"][0], ctx["evento"][0])
        textos.append((f"ALERTA: {nombre}  ({ctx['evento'][1]:.0%})",
                       (12, alto - 32), BLANCO, 18, True))
    else:
        hud.banda(lienzo, alto - 24, alto, 0.60)
        textos.append(("Verde: persona     Rojo: arma detectada     "
                       "Cian a trazos: zona que analiza el modelo de violencia",
                       (10, alto - 21), GRIS, 12, False))

    panel.escribir(lienzo, textos)


def analizar(ruta, cfg, args):
    # OpenCV en Windows falla al escribir en rutas con acentos o caracteres raros
    nombre = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(ruta).stem)[:60] or "video"
    dir_vid = cfg.ruta(cfg.salida.dir_videos)
    dir_evt = cfg.ruta(cfg.salida.dir_eventos)
    dir_vid.mkdir(parents=True, exist_ok=True)
    dir_evt.mkdir(parents=True, exist_ok=True)

    # Se borran los eventos de una corrida anterior de ESTE video: como los
    # nombres son deterministas, volver a analizarlo reemplaza sus resultados
    # en vez de acumular duplicados.
    for viejo in dir_evt.glob(f"{nombre}__*"):
        try:
            viejo.unlink()
        except OSError:
            pass

    fuente = FuenteVideo(
        ruta,
        fps_base=cfg.fuente.fps_base,
        ancho_buffer=cfg.fuente.ancho_buffer,
        buffer_segundos=cfg.fuente.buffer_segundos,
        inicio=args.inicio,
        fin=args.fin,
    )

    print(f"\n>> {Path(ruta).name}  ({fuente.ancho}x{fuente.alto} @ {fuente.fps_nativo:.1f} fps"
          f"{f', {fuente.duracion:.1f} s' if fuente.duracion else ''})")

    personas = DetectorPersonas(cfg)
    armas = DetectorArmas(cfg)
    violencia = DetectorViolencia(cfg) if cfg.violencia.habilitado else None
    cinem = Cinematica(cfg)
    postura = AnalizadorPose(cfg)
    cerebro = Cerebro(cfg)
    evidencia = Evidencia(cfg, fuente, dir_evt)
    panel = hud.Panel()

    escritor = None
    filas = []
    eventos = []
    ocupacion = {"S0": 0.0, "S1": 0.0, "S2": 0.0}

    per_vig = 1.0 / cfg.personas.vigilia.fps
    per_act = 1.0 / cfg.personas.activo.fps
    per_lat = cfg.armas.latido.periodo_seg
    per_arm = 1.0 / cfg.armas.activo.fps
    pc_full = cfg.armas.activo.get("pasada_completa", {})
    full_activo = bool(pc_full.get("activo", False))
    per_full = float(pc_full.get("periodo_seg", 2.0))
    imgsz_full = int(pc_full.get("imgsz", 960))
    per_viol_s2 = cfg.violencia.periodo_seg
    per_viol_s1 = cfg.violencia.get("periodo_s1_seg", per_viol_s2 * 2)
    per_fus = cfg.fusion.evaluar_cada_seg
    n_viol = cfg.violencia.frames_secuencia
    paso_viol = cfg.violencia.stride_sobre_base / cfg.fuente.fps_base

    per_zona = 1.0 / float(cfg.armas.get("seguimiento", {}).get("fps", 2))
    prox_per = prox_arm = prox_viol = prox_fus = prox_full = prox_zona = -1.0
    conteo_vigilia = 0
    p_centro = p_personas = None
    n_eval_viol = 0
    p_viol_vistas = []
    ultimo_evento = None
    t_ult = None
    t0_reloj = time.perf_counter()
    prox_aviso = 0.0

    for m in fuente:
        t = m.t
        if t_ult is not None:
            ocupacion[cerebro.estado] += t - t_ult
        t_ult = t

        # ---------------------------------------------------------- personas
        if cerebro.estado == "S0":
            if t >= prox_per:
                prox_per = t + per_vig
                conteo_vigilia = personas.contar(m.completo)
                antes = cerebro.estado
                cerebro.observar_vigilia(t, conteo_vigilia)
                if cerebro.estado == "S1" and antes == "S0":
                    personas.reiniciar_tracker()
                    cinem.reiniciar()
                    postura.reiniciar()
                    prox_per = t            # pasar ya a la cadencia activa
        else:
            if t >= prox_per:
                prox_per = t + per_act
                tracks = personas.seguir(m.completo, t)
                cinem.actualizar(tracks, t)
                postura.actualizar(tracks, t)
                cerebro.observar_actividad(t, personas.people_count(t))

        # ------------------------------------------------------------- armas
        if cerebro.estado == "S0":
            if cfg.armas.latido.activo and t >= prox_arm:
                prox_arm = t + per_lat
                armas.latido(m.completo, t)
        else:
            if t >= prox_arm:
                prox_arm = t + per_arm
                armas.sobre_personas(m.completo, personas.tracks_actuales, t)
            # Confirmacion dirigida (con TTA): cadencia propia, mas lenta,
            # porque es la ruta cara.
            if t >= prox_zona:
                prox_zona = t + per_zona
                armas.sobre_zonas(m.completo, t)
            # Segunda escala. Se omite mientras haya zonas de confirmacion
            # activas: esas ya cubren la region sospechosa con mas aumento.
            if full_activo and t >= prox_full and not armas.hay_zonas(t):
                prox_full = t + per_full
                armas.latido(m.completo, t, imgsz=imgsz_full)

        # ------------------------------------------- senales -> S1 <-> S2 ---
        if cerebro.estado != "S0":
            cerebro.observar_senales(
                t, armas.hay_alguna(t, 2.0), cinem.senales(t), postura.senales(t)
            )

        # --------------------------------------------------------- violencia
        # Corre en S1 ademas de S2: si solo corriera en S2 quedaria detras de la
        # compuerta "arma o cinematica", y cuando las dos fallan el sistema se
        # queda mudo (caso Video5).
        if violencia and cerebro.estado in ("S1", "S2") and t >= prox_viol:
            cuadros = fuente.muestrear(t, n_viol, paso_viol)
            if cuadros:
                prox_viol = t + (per_viol_s2 if cerebro.estado == "S2" else per_viol_s1)
                caja = personas.caja_grupo()
                caja_b = fuente.a_buffer(caja) if caja else None
                r = violencia.evaluar(cuadros, caja_b)
                p_centro, p_personas = r["centro"], r["personas"]
                cerebro.registrar_violencia(t, violencia.p_decision(r))
                n_eval_viol += 1
                p_viol_vistas.append(violencia.p_decision(r))

        # ------------------------------------------------------------ fusion
        pc = conteo_vigilia if cerebro.estado == "S0" else personas.people_count(t)
        sost, conf_arma, n_dets = armas.sostenida(t)

        if t >= prox_fus:
            prox_fus = t + per_fus
            ev = cerebro.evaluar(t, pc, sost, conf_arma, personas.caja_grupo())
            if ev is not None:
                ev.origen = nombre
                ev.evidence_url = str(evidencia.solicitar(ev))
                ev.guardar(dir_evt)
                eventos.append(ev)
                ultimo_evento = (ev.event_type, ev.confidence)
                print(f"   [{_reloj(t)}] {EVENTO_ES.get(ev.event_type, ev.event_type):<34s}"
                      f" confianza {ev.confidence:.0%}   personas {ev.people_count}"
                      f"   atacantes {ev.atacantes}   [{ev.event_type}]")

            s = cinem.senales(t)
            sp = postura.senales(t)
            filas.append({
                "t": round(t, 2), "estado": cerebro.estado,
                "people_count": pc, "atacantes": max(0, pc - cerebro.atendientes),
                "arma_sostenida": int(sost), "arma_conf": round(conf_arma, 4), "arma_dets": n_dets,
                "p_fight_centro": None if p_centro is None else round(p_centro, 4),
                "p_fight_personas": None if p_personas is None else round(p_personas, 4),
                "cin_proximidad": int(s["proximidad"]), "cin_agitacion": int(s["agitacion"]),
                "cin_aspecto": int(s["aspecto"]), "cin_n": s["n_activas"],
                "cin_vel_max": round(s["vel_max"], 3),
                "cin_umbral_vel": round(s["umbral_vel"], 3),
                "pose_apuntar": int(sp["apuntar"]), "pose_manos_arriba": int(sp["manos_arriba"]),
                "pose_en_suelo": int(sp["en_suelo"]), "pose_amenaza": int(sp["amenaza"]),
                "pose_n_apuntando": sp["n_apuntando"],
                "pose_n_manos_arriba": sp["n_manos_arriba"],
                "evento": ev.event_type if ev is not None else "",
            })

        # ---------------------------------------------------------- anotacion
        if cfg.salida.guardar_video_anotado and not args.sin_video:
            if escritor is None:
                destino = dir_vid / f"{nombre}_anotado.mp4"
                escritor = cv2.VideoWriter(
                    str(destino), cv2.VideoWriter_fourcc(*"mp4v"),
                    cfg.fuente.fps_base, (fuente.ancho_b, fuente.alto_b),
                )
            lienzo = m.pequeno.copy()
            # en S0 no hay tracker: no dibujar cajas viejas de un S1 anterior
            visibles = personas.tracks_actuales if cerebro.estado != "S0" else []
            region_viol = None
            if violencia and cerebro.estado != "S0":
                cg = personas.caja_grupo()
                region_viol = violencia.region_vista(
                    m.pequeno.shape, fuente.a_buffer(cg) if cg else None
                )
            dibujar(lienzo, panel, fuente, cerebro.estado, visibles, armas.para_dibujar(),
                    armas.zonas if cerebro.estado != "S0" else [], region_viol, {
                        "t": t, "pc": pc, "atacantes": max(0, pc - cerebro.atendientes),
                        "arma_conf": conf_arma, "arma_dets": n_dets, "arma_sost": sost,
                        "p_centro": p_centro, "p_personas": p_personas,
                        "umbral_viol": cerebro.umbral_violencia(),
                        "trazas": armas.trazas_confirmadas() if cerebro.estado != "S0" else [],
                        "posturas": postura.posturas if cerebro.estado != "S0" else [],
                        "pose": postura.senales(t) if cerebro.estado != "S0" else {},
                        "cin": cinem.senales(t) if cerebro.estado != "S0" else {},
                        "evento": ultimo_evento,
                    })
            escritor.write(lienzo)

        evidencia.procesar(t)

        if t >= prox_aviso:
            prox_aviso = t + 10.0
            print(f"   ... {t:6.1f}s  estado={cerebro.estado}  personas={pc}", flush=True)

    # ------------------------------------------------------------- cierre ---
    # Evaluacion final. En un archivo, el ultimo tramo no debe descartarse:
    # en Video5 la violencia cruzaba el umbral 0.3 s antes de que acabara el
    # clip, despues de la ultima ventana de decision. En un flujo continuo esto
    # es inocuo; en analisis de archivo evita perder el final del evento.
    if t_ult is not None:
        sost_f, conf_f, _ = armas.sostenida(t_ult)
        pc_f = conteo_vigilia if cerebro.estado == "S0" else personas.people_count(t_ult)
        ev = cerebro.evaluar(t_ult, pc_f, sost_f, conf_f, personas.caja_grupo())
        if ev is not None:
            ev.origen = nombre
            ev.evidence_url = str(evidencia.solicitar(ev))
            ev.guardar(dir_evt)
            eventos.append(ev)
            print(f"   [{_reloj(t_ult)}] {EVENTO_ES.get(ev.event_type, ev.event_type):<34s}"
                  f" confianza {ev.confidence:.0%}   personas {ev.people_count}"
                  f"   atacantes {ev.atacantes}   [{ev.event_type}]  (cierre)")

    evidencia.cerrar()
    if escritor is not None:
        escritor.release()

    ruta_csv = dir_vid / f"{nombre}_ventanas.csv"
    if cfg.salida.guardar_csv and filas:
        with open(ruta_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNAS)
            w.writeheader()
            w.writerows(filas)

    reloj = time.perf_counter() - t0_reloj
    dur = t_ult or 0.0
    total_ocup = sum(ocupacion.values()) or 1.0

    print(f"\n   --- {Path(ruta).name} ---")
    print(f"   video {dur:.1f} s  procesado en {reloj:.1f} s  "
          f"({dur / reloj:.2f}x tiempo real)")
    print("   ocupacion: " + "  ".join(
        f"{k} {v / total_ocup * 100:4.1f}%" for k, v in ocupacion.items()))
    print(f"   transiciones: {len(cerebro.transiciones)}   id_switches: {personas.id_switches}")

    # Diagnostico: por que disparo (o por que no). Sin esto, "0 eventos" es
    # indistinguible de "el pipeline no vio nada".
    if filas:
        n = len(filas)
        con_det = sum(1 for f in filas if f["arma_dets"] > 0)
        con_sost = sum(1 for f in filas if f["arma_sostenida"])
        max_arma = max((f["arma_conf"] for f in filas), default=0.0)
        umbral = cerebro.umbral_violencia()
        sobre = sum(1 for p in p_viol_vistas if p >= umbral)
        cin1 = sum(1 for f in filas if f["cin_n"] >= 1)
        cin2 = sum(1 for f in filas if f["cin_n"] >= 2)
        print(f"   diagnostico sobre {n} ventanas de decision:")
        print(f"      armas     : {con_det:3d} ventanas con deteccion, {con_sost:3d} sostenida, "
              f"conf max {max_arma:.0%}")
        if p_viol_vistas:
            print(f"      violencia : {n_eval_viol:3d} evaluaciones reales, {sobre:3d} sobre el "
                  f"umbral {umbral:.2f}, p max {max(p_viol_vistas):.0%}, "
                  f"p min {min(p_viol_vistas):.0%}")
        else:
            evaluable = max(0.0, dur - violencia.ventana_segundos()) if violencia else 0.0
            print(f"      violencia :   0 evaluaciones -- el modelo necesita "
                  f"{violencia.ventana_segundos():.1f} s de historial" if violencia
                  else "      violencia : desactivada")
            if violencia and evaluable <= 0:
                print(f"                  el video dura {dur:.1f} s: NO alcanza para una "
                      f"sola ventana")
        print(f"      cinematica: {cin1:3d} ventanas con >=1 senal, {cin2:3d} con >=2 "
              f"(>=2 es lo que alerta)")
    print(f"   eventos: {len(eventos)}")
    if filas:
        print(f"   csv: {ruta_csv}")
    if escritor is not None:
        print(f"   video anotado: {dir_vid / f'{nombre}_anotado.mp4'}")
    for p in evidencia.escritos:
        print(f"   evidencia: {p}")

    return eventos


def main():
    ap = argparse.ArgumentParser(description="Agente de deteccion VecindApp Seguro")
    ap.add_argument("--video", help="ruta a un .mp4 (por defecto: todos los de videos/entrada/)")
    ap.add_argument("--config", default=str(RAIZ / "config.yaml"))
    ap.add_argument("--inicio", type=float, default=0.0, help="segundo inicial")
    ap.add_argument("--fin", type=float, default=None, help="segundo final")
    ap.add_argument("--sin-video", action="store_true", help="no generar el video anotado")
    args = ap.parse_args()

    cfg = cfg_mod.cargar(args.config)

    if args.video:
        videos = [Path(args.video)]
    else:
        entrada = cfg.ruta("videos/entrada")
        # Los DVR suelen exportar en .avi o .mov, no solo .mp4
        videos = sorted(
            p for p in entrada.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONES
        )

    if not videos:
        print(f"No hay videos. Deja tus archivos en {cfg.ruta('videos/entrada')}")
        print("Formatos aceptados: " + " ".join(EXTENSIONES))
        return

    print(f"Videos a analizar: {len(videos)}")
    total, fallidos = 0, 0
    for v in videos:
        if not Path(v).exists():
            print(f"  no existe: {v}")
            fallidos += 1
            continue
        try:
            total += len(analizar(v, cfg, args))
        except Exception as exc:                      # un video roto no corta el lote
            fallidos += 1
            print(f"   ERROR en {Path(v).name}: {exc}")

    print(f"\n=== {total} evento(s) en {len(videos) - fallidos} video(s) ===")
    if fallidos:
        print(f"    {fallidos} video(s) no se pudieron procesar")
    print(f"Eventos y clips en: {cfg.ruta(cfg.salida.dir_eventos)}")


if __name__ == "__main__":
    main()

