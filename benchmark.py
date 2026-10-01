"""
Fase 0 - Verificacion y medicion.

1. Comprueba que los tres modelos cargan en un mismo proceso.
2. Mide la latencia real de cada uno en ESTA maquina.
3. Mide el costo de decodificacion sobre un video real.
4. Calcula el presupuesto por estado (S0/S1/S2) leyendo config.yaml.

Uso:  python benchmark.py
"""

import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

RAIZ = Path(__file__).parent
CFG = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))

SEP = "=" * 74


def titulo(t):
    print("\n" + SEP)
    print("  " + t)
    print(SEP)


def medir(fn, repeticiones=8, calentamiento=2):
    for _ in range(calentamiento):
        fn()
    muestras = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        fn()
        muestras.append(time.perf_counter() - t0)
    muestras.sort()
    return statistics.median(muestras) * 1000


# ---------------------------------------------------------------- entorno ---
titulo("ENTORNO")
try:
    import cv2

    cv2_ver = cv2.__version__
except ImportError:
    cv2 = None
    cv2_ver = "NO INSTALADO"

hilos = torch.get_num_threads()
print(f"  Python       : {sys.version.split()[0]}")
print(f"  Plataforma   : {platform.processor() or platform.machine()}")
print(f"  torch        : {torch.__version__}   (CUDA: {torch.cuda.is_available()})")
print(f"  opencv       : {cv2_ver}")
print(f"  Hilos torch  : {hilos}")

try:
    from ultralytics import YOLO
    import ultralytics

    print(f"  ultralytics  : {ultralytics.__version__}")
except ImportError:
    print("  ultralytics  : NO INSTALADO -- abortando")
    sys.exit(1)

import torchvision

print(f"  torchvision  : {torchvision.__version__}")

TAMANOS = [320, 416, 512, 640]
res = {"personas": {}, "armas": {}, "violencia": None, "decode": None}

# --------------------------------------------------------------- personas ---
titulo("1/4  PERSONAS  -  YOLOv8n (COCO)")
mp_ = RAIZ / CFG["personas"]["modelo"]
m_personas = YOLO(str(mp_))
print(f"  Archivo : {mp_.name}  ({mp_.stat().st_size / 1e6:.1f} MB)")
print(f"  Clases  : {len(m_personas.names)}  ->  clase 0 = '{m_personas.names[0]}'")

marco = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
for s in TAMANOS:
    ms = medir(lambda s=s: m_personas.predict(marco, imgsz=s, conf=0.4, classes=[0], verbose=False))
    res["personas"][s] = ms
    print(f"    @{s:<4} : {ms:7.1f} ms   ({1000 / ms:5.1f} fps maximos)")

# ------------------------------------------------------------------ armas ---
titulo("2/4  ARMAS  -  YOLOv8s (fine-tuned)")
ma_ = RAIZ / CFG["armas"]["modelo"]
m_armas = YOLO(str(ma_))
print(f"  Archivo : {ma_.name}  ({ma_.stat().st_size / 1e6:.1f} MB)")
print(f"  Clases  : {m_armas.names}")
if len(m_armas.names) == 1:
    print("  NOTA    : UNA sola clase. No hay 'pistol'/'knife' por separado,")
    print("            asi que no se pueden aplicar umbrales por clase.")

recorte = np.random.randint(0, 255, (480, 260, 3), dtype=np.uint8)  # caja de persona tipica
for s in TAMANOS:
    fuente = recorte if s <= 416 else marco
    ms = medir(lambda s=s, f=fuente: m_armas.predict(f, imgsz=s, conf=0.45, verbose=False))
    res["armas"][s] = ms
    print(f"    @{s:<4} : {ms:7.1f} ms   ({1000 / ms:5.1f} fps maximos)")

# -------------------------------------------------------------- violencia ---
titulo("3/4  VIOLENCIA  -  MC3-18")
mv_ = RAIZ / CFG["violencia"]["modelo"]
print(f"  Archivo : {mv_.name}  ({mv_.stat().st_size / 1e6:.1f} MB)")

# weights=None: el repo original usa pretrained=True, que descarga ~50 MB de
# pesos Kinetics en CADA arranque solo para sobreescribirlos aqui abajo.
m_viol = torchvision.models.video.mc3_18(weights=None)
m_viol.fc = torch.nn.Linear(m_viol.fc.in_features, 2)
estado = torch.load(str(mv_), map_location="cpu", weights_only=True)
faltan, sobran = m_viol.load_state_dict(estado, strict=False)
m_viol.eval()
print(f"  Tensores: {len(estado)}   faltantes: {len(faltan)}   inesperados: {len(sobran)}")
if faltan or sobran:
    print("  AVISO   : el state_dict no encaja del todo con mc3_18")

n_frames = CFG["violencia"]["frames_secuencia"]
clip = torch.randn(1, 3, n_frames, 112, 112)
with torch.no_grad():
    ms_viol = medir(lambda: m_viol(clip), repeticiones=6)
res["violencia"] = ms_viol

stride = CFG["violencia"]["stride_sobre_base"]
fps_base = CFG["fuente"]["fps_base"]
ventana = n_frames * stride / fps_base
print(f"    {n_frames} frames @ 112x112 : {ms_viol:7.1f} ms")
print(f"    Ventana temporal          : {ventana:.2f} s  (stride {stride} sobre {fps_base} fps)")
print(f"    Entrenamiento original    : ~4.8 s  ->  {'COINCIDE' if abs(ventana - 4.8) < 0.5 else 'NO COINCIDE'}")

# ----------------------------------------------------------- decodificacion --
titulo("4/4  DECODIFICACION")
videos = sorted((RAIZ / "videos" / "entrada").glob("*.mp4"))
if cv2 and videos:
    v = videos[0]
    cap = cv2.VideoCapture(str(v))
    fps_nat = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video   : {v.name}  ({w}x{h} @ {fps_nat:.1f} fps)")

    t0 = time.perf_counter()
    n = 0
    while n < 120 and cap.grab():
        cap.retrieve()
        n += 1
    ms_full = (time.perf_counter() - t0) / max(n, 1) * 1000
    cap.release()

    cap = cv2.VideoCapture(str(v))
    t0 = time.perf_counter()
    n2 = 0
    while n2 < 120 and cap.grab():
        n2 += 1
    ms_grab = (time.perf_counter() - t0) / max(n2, 1) * 1000
    cap.release()

    res["grab"] = ms_grab
    res["retrieve"] = max(ms_full - ms_grab, 0.0)
    res["fps_nativo"] = fps_nat if fps_nat and fps_nat > 0 else 30.0
    res["px"] = w * h
    print(f"    grab + retrieve : {ms_full:6.2f} ms/frame")
    print(f"    solo grab       : {ms_grab:6.2f} ms/frame")
    print(f"    conversion BGR  : {res['retrieve']:6.2f} ms/frame")
    print()
    print("    NOTA: desde una fuente de N fps hay que hacer grab de los N frames")
    print("          (H.264 encadena los cuadros); solo se hace retrieve de los que")
    print("          se procesan. El costo NO es fps_base x (grab+retrieve).")
else:
    res["grab"], res["retrieve"] = 2.3, 5.1
    res["fps_nativo"], res["px"] = 30.0, 1920 * 1080
    print("  Sin video en videos/entrada/ -> se asumen valores tipicos de 1080p30")

# ------------------------------------------------------------ presupuesto ---
titulo("PRESUPUESTO POR ESTADO  (segundos de computo por segundo de video)")

cp, ca, cv_ = CFG["personas"], CFG["armas"], CFG["violencia"]


def yolo_ms(tabla, imgsz):
    return tabla.get(imgsz) or tabla[min(tabla, key=lambda k: abs(k - imgsz))]


fps_nat = res["fps_nativo"]
dec = (fps_nat * res["grab"] + fps_base * res["retrieve"]) / 1000

s0_pers = cp["vigilia"]["fps"] * yolo_ms(res["personas"], cp["vigilia"]["imgsz"]) / 1000
s0_lat = (
    yolo_ms(res["armas"], ca["latido"]["imgsz"]) / 1000 / ca["latido"]["periodo_seg"]
    if ca["latido"]["activo"]
    else 0.0
)
S0 = dec + s0_pers + s0_lat

s1_pers = cp["activo"]["fps"] * yolo_ms(res["personas"], cp["activo"]["imgsz"]) / 1000
n_crops = 2
s1_arm = ca["activo"]["fps"] * n_crops * yolo_ms(res["armas"], ca["activo"]["imgsz"]) / 1000
S1 = dec + s1_pers + s1_arm

s2_viol = res["violencia"] / 1000 / cv_["periodo_seg"] if cv_["habilitado"] else 0.0
S2 = S1 + s2_viol

print(f"  S0 VIGILIA      people_count <= 1")
print(f"     decode: {fps_nat:.0f} grab + {fps_base} retrieve     {dec:6.3f}")
print(f"     personas {cp['vigilia']['fps']} fps @{cp['vigilia']['imgsz']} sin tracker   {s0_pers:6.3f}")
print(f"     latido de armas cada {ca['latido']['periodo_seg']} s          {s0_lat:6.3f}")
print(f"                                   TOTAL  {S0:6.3f} s/s   ({S0 * 100:4.0f}%)")

print(f"\n  S1 ACTIVIDAD    people_count >= 2")
print(f"     decodificacion                    {dec:6.3f}")
print(f"     personas {cp['activo']['fps']} fps @{cp['activo']['imgsz']} + ByteTrack  {s1_pers:6.3f}")
print(f"     armas {ca['activo']['fps']} fps @{ca['activo']['imgsz']} x {n_crops} recortes    {s1_arm:6.3f}")
print(f"                                   TOTAL  {S1:6.3f} s/s   ({S1 * 100:4.0f}%)")

print(f"\n  S2 EVALUACION   arma o cinematica")
print(f"     todo lo de S1                     {S1:6.3f}")
print(f"     MC3-18 cada {cv_['periodo_seg']} s                {s2_viol:6.3f}")
print(f"                                   TOTAL  {S2:6.3f} s/s   ({S2 * 100:4.0f}%)")

ocup = (0.65, 0.33, 0.02)
prom = ocup[0] * S0 + ocup[1] * S1 + ocup[2] * S2
print(f"\n  Promedio ponderado (65% S0 / 33% S1 / 2% S2)  =  {prom:.3f} s/s  ({prom * 100:.0f}%)")

sin_compuertas = (
    dec
    + 5 * yolo_ms(res["personas"], 640) / 1000
    + 5 * yolo_ms(res["armas"], 640) / 1000
    + (res["violencia"] / 1000 if cv_["habilitado"] else 0)
)
print(f"  Sin compuertas (los 3 a 5 fps @640)           =  {sin_compuertas:.3f} s/s")
print(f"  Mejora de la cascada                          =  {sin_compuertas / prom:.1f}x")

titulo("VEREDICTO")
print(f"  Los tres modelos cargan en el mismo proceso: OK")
for etiqueta, val in (("S0", S0), ("S1", S1), ("S2", S2)):
    estado_txt = "OK" if val < 1.0 else "SOBRE TIEMPO REAL (se absorbe con el buffer)"
    print(f"  {etiqueta}: {val:.3f} s/s  -> {estado_txt}")
print(f"\n  Esta maquina procesa 1 camara con {(1 - prom) * 100:.0f}% de margen promedio.")
print(f"  Mini PC de 4 nucleos (~2.5x mas lento): ~{prom * 2.5:.2f} s/s promedio.")

# La decodificacion es proporcional a los pixeles: bajar de 1080p a 720p la
# abarata 2.25x. Es la palanca mas grande sobre el costo de S0.
px_720 = 1280 * 720
if res["px"] > px_720:
    dec720 = dec * px_720 / res["px"]
    ahorro = dec - dec720
    print(f"\n  La fuente medida es {res['px'] / 1e6:.1f} MP. Con un substream 720p:")
    print(f"     decodificacion  {dec:.3f} -> {dec720:.3f} s/s   (-{ahorro:.3f})")
    print(f"     S0              {S0:.3f} -> {S0 - ahorro:.3f} s/s")
    print(f"     promedio        {prom:.3f} -> {prom - ahorro * 0.98:.3f} s/s")
    print("  RESUMEN seccion 12 ya recomienda el substream de baja resolucion;")
    print("  ademas de ancho de banda, es la mayor palanca sobre el costo de vigilia.")
print(SEP)
