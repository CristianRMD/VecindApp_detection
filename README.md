# VecindApp Seguro — Agente local de detección

Consolidación de los tres modelos validados en los tickets ME (armas, personas, violencia)
en un solo pipeline con compuertas por estado.

> **Estado: Fases 0–2 completas.** Pipeline funcionando de extremo a extremo.
> Falta la Fase 3: calibrar los umbrales contra video real de bodega.

**Licencia:** AGPL-3.0 (ver [`LICENSE`](LICENSE)) — impuesta por la dependencia de
Ultralytics, cuya licencia declaran los propios archivos de pesos.
**Antes de reutilizar este repositorio**, lee [`AVISOS.md`](AVISOS.md): documenta la
procedencia de los modelos, la de los videos de prueba incluidos y las implicaciones
de licenciamiento para un despliegue comercial.

**Documentación:** [`MANUAL.txt`](MANUAL.txt) para uso operativo · este README como
bitácora técnica · `Entregable - Documento Tecnico de Integracion.docx` como
documento formal.

---

## Cómo se usa

```bash
# 1. Activar el entorno
.venv\Scripts\activate

# 2. Verificar que todo carga y medir la máquina
python benchmark.py

# 3. (Fase 2) Analizar un video
python analizar.py --video videos/entrada/mi_video.mp4
```

**Sube tus `.mp4` de prueba a `videos/entrada/`.**

Conviene subir dos tipos:
- **Incidentes** — atracos, forcejeos, armas visibles.
- **Operación normal** — atención, cobro, clientes entrando.

Los normales son los que permiten medir **falsos positivos**, que es el conjunto
que pide RESUMEN §6 y que todavía no existe en el proyecto.

### Qué produce cada corrida

| Salida | Contenido |
|---|---|
| `videos/salida/<nombre>_anotado.mp4` | cajas de persona con ID, cajas de arma, y un HUD con estado, conteo, confianza del arma, `p_fight` de ambas variantes y qué señales cinemáticas están activas |
| `videos/salida/<nombre>_ventanas.csv` | una fila por segundo con **todas** las señales, disparen o no. Es el archivo de calibración |
| `eventos/<event_id>.json` | evento según el contrato de RESUMEN §13 |
| `eventos/<event_id>.mp4` | clip de evidencia: 10 s antes y 5 s después |

El CSV es lo importante para la Fase 3: registra `p_fight_centro` y
`p_fight_personas` **en paralelo**, junto con las señales cinemáticas, en cada
ventana. Eso permite comparar head-to-head qué detecta cada capa sobre tu propio
video — el dato que hoy no existe en ninguno de los tres informes.

Por consola, al terminar cada video se imprime la velocidad de proceso
(× tiempo real), la ocupación de cada estado y el número de transiciones.

---

## Qué se extrajo de cada repo

| Origen | Archivo | Destino | Tamaño |
|---|---|---|---|
| `Weapon-Detection-System-main/weapon-detection-system/backend/best.pt` | YOLOv8s fine-tuned | `models/weapon_yolov8s.pt` | 21.5 MB |
| `Analisis_ME_proyecto/yolov8n.pt` | YOLOv8n oficial COCO | `models/person_yolov8n.pt` | 6.2 MB |
| `ProyectoFight-main/Models/model_16_m3_0.8888.pth` | MC3-18 fine-tuned | `models/violence_mc3_18.pth` | 43.9 MB |

Los tres repos originales quedaron **intactos** (copia, no movimiento).

**No se trajo:** las dos web apps Next.js, `app.py` (Streamlit), los 5 notebooks,
`infer.py`, `setup.py`, el paquete pip, el `dataset/` de 300 clips, ni el
`best.pt` de la raíz del repo de armas — que está **roto** (2 bytes: `0D 0A`,
el origen del `UnpicklingError: invalid load key, '\x0d'`).

---

## Procedencia y métricas reales de cada modelo

Ninguno de los tres se entrenó en este proyecto. Esto es lo que se verificó
leyendo los checkpoints y los notebooks de origen.

### Personas — `person_yolov8n.pt`

```
Entrenado por : Ultralytics
Fecha         : 2022-12-30 · ultralytics 8.0.0.dev0
Datos         : COCO (80 clases, ~118k imágenes, 500 épocas, imgsz 640)
Licencia      : AGPL-3.0  ← embebida en el propio archivo de pesos
```

Sin fine-tuning. Se usa solo la clase `0: person`. Es el componente **más
confiable** de los tres: `person` es la clase más representada de COCO y la
transferencia a cualquier cámara es segura.

**ByteTrack no es una red neuronal** — no tiene pesos ni se entrena. Es filtro
de Kalman + asignación húngara sobre IoU. De ahí la dependencia `lap`.

### Armas — `weapon_yolov8s.pt`

```
Base          : yolov8s.pt (COCO)
Fine-tuning   : 100 épocas · batch 16 · imgsz 640 · Adam lr0 0.001 · cos_lr
Fecha         : 2025-05-18 · ultralytics 8.3.138
Datos         : /content/drive/MyDrive/Colab Notebooks/dataset2/  ← PRIVADO, no disponible
Clases        : {0: 'weapon'}   ← UNA sola clase
```

> ⚠️ El documento `analisis_deteccionarmas.txt` reporta *"2 clases: pistol, knife"*.
> **Este checkpoint no las tiene.** Por tanto la calibración recomendada allí
> (`conf ≥ 0.60` para pistol, `≥ 0.35` para knife) **no es implementable**.
> Hay que resolver qué checkpoint se validó antes de fijar umbrales.

**Métricas reales** (salida de `model.val()`, en `Visualization/` del repo original):

| | |
|---|---|
| mAP@0.5 | **0.797** |
| Precisión | 190/222 = **85.6 %** |
| Recall | 190/257 = **73.9 %** — *se le escapa 1 de cada 4 armas* |
| F1 | 0.794 |
| Matriz | TP 190 · FP 32 · FN 67 |

La curva PR tiene un **codo marcado**: precisión ~1.0 hasta recall ~0.6, luego
se desploma (en recall 0.8, precisión ~0.5). Ese codo es el punto de calibración.

Y esto es sobre su **propio** conjunto de validación, misma distribución que el
entrenamiento. En cámara de bodega peruana será peor.

### Violencia — `violence_mc3_18.pth`

```
Arquitectura  : torchvision.models.video.mc3_18 (11.5 M parámetros, convs 3D)
Base          : Kinetics-400 (400 acciones, ~240k clips)
Fine-tuning   : 30 épocas · SGD lr 0.001 momentum 0.9 · batch 4 · seed 70
Datos         : 300 clips (150 fight / 150 noFight) → 210 train / 45 val / 45 test
Clases        : {0: 'fight', 1: 'noFight'}
```

**El "0.8888" del nombre es la precisión: 40 aciertos de 45 clips.**
Intervalo de confianza 95 % ≈ **[76 %, 95 %]**. Orientativa, no concluyente.

Cuatro problemas documentados que el pipeline corrige o rodea:

1. **El notebook publicado entrena `r2plus1d_18`, no `mc3_18`.** El código que
   produjo este checkpoint **no está en el repo**. No es reentrenable ni auditable.

2. **Desajuste de muestreo temporal.** El entrenamiento toma 16 frames repartidos
   sobre todo el clip (~4.8 s ⇒ ~3.3 fps efectivos). El `infer.py` del repo usa
   `--skip 2`, que da una ventana de **1.07 s** — ~4.5× fuera de distribución.
   → Corregido con `stride_sobre_base: 3` sobre `fps_base: 10`.

3. **No emite confianza.** `PredTopKClass()` descarta la probabilidad. Los números
   `0.9966` / `0.3833` del documento de validación **no son confianzas**: son el
   `print(end-start)` de `infer.py`, es decir **segundos**.
   → Corregido usando la rama con softmax.

4. **Desajuste de objetivo.** Se entrenó para `fight` vs `noFight`. RESUMEN §2
   incluye *"intimidación física o sometimiento"*. Un atraco típico de bodega
   (arma + obediencia, sin pelea) sale `noFight`. El propio documento reporta
   *"arresto policial → noFight"* como **acierto**, cuando para VecindApp es un
   falso negativo. → Por eso existe la capa `cinematica`.

**Además:** de los 16 videos de prueba del documento, 8 salen del `dataset/` del
propio repo (datos de entrenamiento) y los otros 8 son **todos casos negativos**.
Se probaron **cero** videos de violencia externos: la sensibilidad sobre video
nuevo está **sin medir**.

---

## Medidas reales (Fase 0)

`benchmark.py` sobre AMD Ryzen 5 4600H (6 núcleos), 6 hilos torch, torch 2.14.0+cpu,
ultralytics 8.4.166. Latencias por inferencia:

| imgsz | YOLOv8n (personas) | YOLOv8s (armas) |
|---|---|---|
| 320 | 30.0 ms | 45.9 ms |
| 416 | 32.7 ms | 61.2 ms |
| 512 | 37.9 ms | 72.5 ms |
| 640 | 49.2 ms | 107.4 ms |

**MC3-18 (violencia):** 445 ms por clip de 16 frames.
**Decodificación 1080p30:** 1.76 ms `grab` + 6.53 ms conversión BGR.

Dos cosas que salieron de aquí y que no eran obvias:

- **Por debajo de 416 no vale la pena bajar.** YOLOv8n cuesta 30.0 ms @320 contra
  32.7 ms @416 — apenas 9 %. A esos tamaños el costo ya no lo dominan las
  convoluciones sino el preprocesado y el NMS. Así que se usa 416, que detecta
  mejor, por casi el mismo precio.

- **La decodificación es la mitad del costo de vigilia.** Desde una fuente de
  N fps hay que hacer `grab` de los N cuadros (H.264 los encadena), aunque solo
  se convierta a BGR los que se procesan. A 1080p30 eso son 0.118 s/s — más que
  el propio detector de personas. Con un substream 720p baja a 0.052 s/s.
  RESUMEN §12 ya recomienda el flujo secundario de baja resolución por ancho de
  banda; resulta que además es **la mayor palanca sobre el costo de CPU**.

## Arquitectura: cascada con compuertas

Correr los tres modelos sobre cada frame cuesta **1.36 s/s** — por encima de
tiempo real, y sin margen alguno en un mini PC. La solución no es bajar fps
globalmente (a 1 fps ByteTrack deja de asociar cajas y el modelo de violencia
sale de su distribución), sino **encender cada modelo solo cuando hace falta**.

```
         RTSP / .mp4
              │  decodificación ÚNICA @10 fps (muestreo por timestamp)
              ▼
    ┌─────────────────────────────────────────────────┐
    │ S0 · VIGILIA          people_count ≤ 1          │
    │   YOLOv8n @416, 2 fps, SIN tracker              │
    │   + latido de armas cada 5 s                    │
    └──────────────────┬──────────────────────────────┘
                       │ ≥2 personas en 2 de 3 muestras (~1 s)
                       ▼
    ┌─────────────────────────────────────────────────┐
    │ S1 · ACTIVIDAD                                  │
    │   YOLOv8n @416, 5 fps + ByteTrack                │
    │   + armas @320 sobre recortes de persona, 2 fps │
    │   + cinemática (gratis, sale del tracker)       │
    └──────────────────┬──────────────────────────────┘
                       │ arma detectada  O  disparador cinemático
                       ▼
    ┌─────────────────────────────────────────────────┐
    │ S2 · EVALUACIÓN                                 │
    │   + MC3-18 cada 2 s (ventana de 4.8 s)          │
    └──────────────────┬──────────────────────────────┘
                       │ reglas de fusión
                       ▼
             S3 · EVENTO → JSON + clip de evidencia
```

### Presupuesto medido (fuente 1080p30)

| Estado | Ocupación estimada | Costo |
|---|---|---|
| S0 Vigilia | ~65 % | **0.200 s/s** (20 %) |
| S1 Actividad | ~33 % | **0.476 s/s** (48 %) |
| S2 Evaluación | ~2 % | **0.699 s/s** (70 %) |
| **Promedio ponderado** | | **0.301 s/s** (30 %) |

Sin compuertas serían 1.358 s/s → la cascada da **4.5×**. Con substream 720p el
promedio baja a **0.237 s/s**.

Ningún estado supera tiempo real, ni siquiera S2 — queda margen para subir la
frecuencia de MC3-18 a 1 s si la calibración lo justifica.

**Mini PC de 4 núcleos** (~2.5× más lento): ~0.75 s/s promedio. Entra para una
cámara sin GPU, que es justo lo que plantea RESUMEN §12.

### Decisiones de diseño

**En vigilia no hace falta seguimiento, solo contar.** La pregunta en S0 no es
*"¿quién es quién?"* sino *"¿hay más de una persona?"*. Contar cajas funciona a
2 fps; el tracker se enciende solo al entrar en S1. Eso baja el costo de vigilia
4.7× respecto a correr el tracker siempre.

**"Siempre hay 1 persona atendiendo"** se aprovecha como constante
(`atacantes = people_count − 1`), no como modelo espacial aprendido. Se evaluó
auto-calibrar la zona de atención con un mapa de calor y se descartó: **no cambia
la compuerta de cómputo** (ambos diseños activan en `people_count ≥ 2`), y el
único escenario que resolvía — el que atiende salió al fondo — se cubre con el
**latido de armas** por 2 % de cómputo en vez de ~250 líneas y 4 modos de fallo.

**Recortes por persona para el detector de armas.** Un arma de 20 px en el cuadro
completo escalado a 512 queda en 8 px; recortando la caja de la persona queda en
~14 px. Casi 2× de resolución efectiva, y más barato que una pasada a 640. Crítico
dado que el documento de armas reporta cuchillos de 4–10 px a 3 m.

**El pipeline debe ser tiempo real en promedio, no en cada instante.** S2 puede
superar 1.0 s/s; como es ~2 % del tiempo y hay 30 s de buffer circular, el atraso
se recupera al salir del estado y no se pierde evidencia.

---

## Reglas de fusión → evento

Ventana deslizante de 5 s, evaluada cada 1 s. `atacantes = people_count − 1`.

| Condición | atacantes | `event_type` | severity |
|---|---|---|---|
| Arma persistente (3 det. / 2 s), sin violencia | cualquiera | `suspicious_weapon` | media |
| Violencia **o** cinemática sostenida, sin arma | 1 | `physical_violence` | media |
| Arma **+** (violencia o cinemática) | 1 | `possible_robbery` | **alta** |
| Arma **+** (violencia o cinemática) | **≥2** | `possible_multiple_robbery` | **alta** |
| Varias personas, sin arma ni violencia | ≥1 | *(ninguno)* | — |

El JSON sigue el contrato de RESUMEN §13. `confidence` es media ponderada
(`0.6·arma + 0.4·violencia`, configurable); las confianzas individuales se
preservan en `detections[]`.

---

## Riesgo de licencia (pendiente de decisión del equipo)

Los dos repos son MIT, pero **`ultralytics` es AGPL-3.0** — está escrito dentro
del propio `person_yolov8n.pt`. Instalar el agente en el mini PC de cada negocio
es **distribución**, y AGPL obliga a liberar el código del sistema enlazado.

Salidas, en orden de preferencia:

1. **Exportar a ONNX y ejecutar con `onnxruntime` (MIT)**, sin empaquetar
   `ultralytics` en el agente. Resuelve además el rendimiento (2–3× en CPU): dos
   problemas con un cambio.
2. Licencia comercial de Ultralytics.
3. Migrar a un detector Apache-2.0 (RT-DETR, DAMO-YOLO) — implica reentrenar.

---

## Estructura

```
VecindApp_Deteccion/
├── config.yaml           todos los umbrales y fps en un solo archivo
├── benchmark.py          verifica carga + mide la máquina + calcula presupuesto
├── analizar.py           (Fase 2) CLI principal
├── models/               los 3 checkpoints
├── src/                  (Fases 1–2) módulos del pipeline
├── videos/entrada/       ← TUS .mp4 DE PRUEBA
├── videos/salida/        videos anotados
└── eventos/              JSON + clips de evidencia
```

## Fases

| Fase | Contenido | Estado |
|---|---|---|
| **0** | Consolidación, entorno, benchmark | ✅ |
| **1** | Los 3 detectores + cinemática | ✅ |
| **2** | Fuente única, máquina de estados, fusión, evidencia | ✅ |
| **3** | Calibración con los videos de `videos/entrada/` | ⬜ |
| **4** | RTSP, reconexión, envío a la API, export ONNX | ⬜ |

## Calibración: lo que ya se ajustó y por qué

Dos reglas se corrigieron al correr el pipeline sobre video real. Quedan
documentadas porque ambas son trampas fáciles de repetir:

**La proximidad sola no puede alertar.** La primera versión emitía
`physical_violence` en cuanto dos personas estaban cerca de forma sostenida. En
una bodega el cliente **siempre** está cerca del que atiende: es el patrón
normal, no violencia. Ahora la cinemática usa dos umbrales distintos:

- **≥1 señal → escalar a S2.** Encender el modelo de violencia cuesta CPU, no
  una alerta; conviene ser permisivo.
- **≥2 señales → emitir evento.** `proximidad + agitación` = forcejeo;
  `proximidad + aspecto` = sometimiento; `agitación + aspecto` = movimiento
  violento con cambio de postura.

**La persistencia del arma debe ajustarse al muestreo.** El informe de armas pide
"3 cuadros consecutivos". Aplicado literalmente a 2 fps en una ventana de 2 s son
3 de 4 muestras — con un recall de 73.9 % eso casi nunca se cumple, y en la
primera corrida `arma_sostenida` nunca se activó pese a detecciones de conf 0.61.
Ahora son 3 detecciones en 3 s muestreando a 3 fps (3 de 9): sigue descartando el
parpadeo de un solo cuadro, pero es alcanzable.

**El conteo que mandan las reglas es el máximo de la ventana de decisión**, no el
instantáneo: durante un forcejeo las personas se ocluyen entre sí y el conteo cae
justo en el momento del evento.

## Resultados sobre los 5 videos de prueba (30-09-2026)

Cinco clips de noticieros peruanos con asaltos reales a comercios.

| Video | Escenario | Resultado | |
|---|---|---|---|
| 1 | Tienda de celulares, varios asaltantes destruyen mostrador | `physical_violence` 97 % | ✅ |
| 2 | Botica, delincuente finge estar enfermo y somete al que atiende | `possible_robbery` 79 % | ✅ |
| 3 | Restaurante, varios asaltantes | `physical_violence` 96 % | ✅ |
| 4 | Pollería | arma → `possible_multiple_robbery` 80 % | ✅ |
| 5 | Tienda de tecnología, **pistola apuntando a un empleado** | arma → `possible_robbery` 66 % | ✅ |

**5 de 5 alertan.** En 3 de los 5 el arma llega a confirmarse; en Video1 y
Video3 la alerta sale por violencia (ver más abajo). Velocidad: 0.80× a 1.79×
tiempo real — los lentos son los que pasan 75-93 % en S2, proporción que en
operación real sería ~2 %.

### Las tres mejoras y qué recuperó cada una

**1. Muestreo denso de violencia en S2** (`periodo_seg: 2.0 → 1.0`). La señal es
muy sensible a dónde cae la ventana de 4.8 s: en Video5 un barrido a 1 Hz veía
`p_fight = 0.977` mientras el pipeline a 0.5 Hz se quedaba en 0.75. **La ventana
buena existía y la saltábamos.** Tras el cambio, Video5 llega a 98 %.

**2. Confirmación dirigida del arma** (*detect-then-verify*). Tras cualquier
detección, por débil que sea, se abre una zona sobre esa región y se la
re-inspecciona ampliada durante 4 s. Un arma genuinamente visible solo se
detecta en el 14-29 % de los cuadros, así que el detector la ve y la pierde;
esto convierte un acierto esporádico en una traza.

**3. Umbral relativo de violencia.** El modelo dio p50 = 0.977 en 49 ventanas:
un número fijo no sirve. Pero sí discrimina en relativo (Video2 fue de 0.234 en
calma a 0.977 en la agresión), así que se aprende la línea base de cada cámara y
se alerta por desviación. *Solo se activa tras 40 evaluaciones*, de modo que en
clips cortos no llega a entrar: es una función de operación continua.

**Y una cuarta, descubierta al probar:** evaluación de cierre al final del
archivo. En Video5 la violencia cruzaba el umbral **0.3 s antes de que acabara
el clip**, después de la última ventana de decisión. Sin ese cierre se perdía el
evento entero.

## Detección de armas: qué se midió y qué se cambió

El arma era el punto débil. Se compararon cinco técnicas de inferencia sobre los
tramos donde se sabe que hay un arma visible, midiendo **en qué fracción de los
cuadros aparece** (no la confianza máxima: la persistencia necesita varias
vistas, no una buena).

| Técnica | Video5 | Video3 | Video2 | Coste |
|---|---|---|---|---|
| base (recortes @640) | 3/21 | 2/16 | 4/21 | 1× |
| **TTA (`augment=True`)** | **9/21** | 2/16 | **6/21** | 1.3–2× |
| cuadro completo @1280 | 6/21 | 1/16 | 3/21 | 0.8× |
| mosaico tipo SAHI | 7/21 | **0/16** ⚠️ | 4/21 | 2× |
| mosaico + recortes | 8/21 | 2/16 | 5/21 | 3× |

**TTA gana y nunca perjudica.** El mosaico (teselas a resolución nativa, la
receta estándar para objetos pequeños) **empeora** Video3: trocear destruye el
contexto, y este modelo depende de él. Descartado pese a ser la técnica "de
libro".

### El cambio de fondo: seguir el arma, no contar detecciones

El dato clave estaba en la columna de confianza baja: a ≥0.25 el arma aparece en
el **43–95 %** de los cuadros. Casi siempre produce *algo*; el problema era
separar ese algo del ruido.

Contar "N detecciones en una ventana" es **espacialmente ciego**: dos falsos
positivos en esquinas opuestas cuentan igual que dos vistas del mismo arma. Por
eso se bajó el umbral a 0.20 y las detecciones se enlazan en **trazas** por
cercanía espacial, confirmadas por tres condiciones a la vez (número de
observaciones, confianza media y un pico decente).

No es un intercambio entre sensibilidad y precisión — es mejor en las dos: más
sensible porque aprovecha detecciones débiles que antes se tiraban, más
específica porque el ruido está disperso y no forma trazas coherentes.

El TTA se reserva para las **zonas de confirmación**, no para el barrido amplio:
búsqueda barata, verificación cara y dirigida.

**Resultado:** Video5 pasó de *no detectar nunca el arma* (pico 33 %) a
confirmarla y emitir `possible_robbery`. Los picos de confianza también subieron
en Video1 (45 % → 63 %) y en el video de muestra.

## Detección por pose

YOLOv8n-pose **reemplaza** al detector de personas en S1+ (no se suma): da las
mismas cajas más 17 puntos clave por **+9 % de coste** (23.3 ms contra 21.4 ms).

La idea: una pistola mide 20-40 px sobre CCTV y se pierde en el 71-86 % de los
cuadros. **La postura de apuntar mide cientos de píxeles.** Se puede reconocer
el *acto* de apuntar sin llegar a ver el arma.

| Señal | Geometría | Indicador de RESUMEN §2 |
|---|---|---|
| `apuntar` | brazo recto, muñeca lejos del torso, y la prolongación del brazo pasa por el torso de **otra persona** | amenaza con arma |
| `manos_arriba` | muñeca por encima del hombro, **excluyendo el brazo que apunta** | sometimiento |
| `en_suelo` | torso más horizontal que vertical | caída, persona reducida |

La tercera condición de `apuntar` — que el rayo apunte a alguien — es la que lo
hace específico: estirarse hacia un estante no apunta a nadie.

### Calidad de los puntos clave (medida)

| | altura mediana | conf. hombro | conf. codo | conf. muñeca |
|---|---|---|---|---|
| Video5 | 405 px | 0.97 | 0.91 | 0.86 |
| Video3 | 276 px | 0.98 | 0.91 | 0.78 |
| Video2 | 496 px | 0.89 | 0.77 | 0.70 |
| Video4 | 321 px | 0.96 | 0.74 | 0.48 |

La pose **sí es fiable** a esta resolución. Dato útil: la extensión del brazo por
sí sola discrimina poco (el 34-73 % de los brazos la pasan, porque estar de pie
con el brazo colgando ya es "recto"). Quien filtra de verdad es el alejamiento
del torso y, sobre todo, que el rayo pase por otra persona.

### Resultado

`apuntar` dispara en **Video4 y Video5** — los dos con gesto de apuntar
visible — y **en ninguno de los otros cuatro**. Cero disparos espurios.
Video5 pasó de `possible_robbery` 66 % a **`possible_multiple_robbery` 78 %**.

`manos_arriba` no dispara en ningún video, y es correcto: nadie se rinde con las
manos en alto en estos clips (en Video5 la víctima se queda paralizada tras el
mostrador). La señal queda lista para cuando aparezca.

En el video anotado se ve el esqueleto, y cuando alguien apunta se dibuja una
**flecha magenta** desde su muñeca hacia la persona señalada.

### Lo que sigue sin resolverse

Video1 y Video3 alertan por **violencia**, no por arma: sus detecciones no
alcanzan a formar una traza confirmada (picos de 63 % y 48 %). Se podría seguir
aflojando los umbrales hasta que entren, pero **sería sobreajustar a cinco
ejemplos positivos sin un solo negativo**. El límite real es el recall de 73.9 %
del modelo sobre video CCTV, y la salida es reentrenarlo.

### La capa cinemática no aportó nada

En los 6 videos, `cin_n >= 2` (la condición que alerta) **no se cumplió ni una
vez**. Hubo un pico legítimo (Video2, `vel_max` 1.108 contra un umbral de 0.6)
pero duró una sola muestra y el filtro de sostenimiento lo descartó — que es el
comportamiento correcto, pero deja la capa sin contribuir.

No cuesta nada (sale de los tracks que ya existen), pero hoy es peso muerto.
Necesita video con forcejeo prolongado para calibrarse, o replantearse.

## El modelo de violencia NO está saturado (corrección)

Durante varias iteraciones se sospechó que MC3-18 estaba saturado: sobre 49
ventanas de los cinco clips cortos daba **p50 = 0.977** y el 79.6 % superaba el
umbral. Un clasificador que siempre dice "sí" no aporta nada.

**La sospecha era un artefacto del muestreo.** Los cinco clips son montajes de
noticiero que cortan directo al atraco: son 100 % evento, así que una p alta
constante era lo correcto.

`video6.mp4` (2:18, incluye tramos normales de la botica) lo deja claro:

| | ventanas | p50 | ≥ umbral 0.75 |
|---|---|---|---|
| 5 clips cortos (todo evento) | 49 | 0.977 | 79.6 % |
| **video6 (contenido mixto)** | 133 | **0.113** | **12.8 %** |

Con `p10 = 0.008` y `min = 0.000`, el modelo **discrimina perfectamente** sobre
material tranquilo. El umbral absoluto de 0.75 es razonable, y el umbral
relativo (`umbral_relativo`) queda como refinamiento por cámara, no como rescate.

Lección metodológica: **cinco positivos no son una muestra.** Un solo video con
contenido mixto dijo más sobre el modelo que todos los clips cortos juntos.
