# Avisos legales y de procedencia

Este archivo documenta la licencia del proyecto, el origen de los componentes de
terceros y la naturaleza del material de video incluido en el repositorio.

---

## 1. Licencia del proyecto

Este proyecto se distribuye bajo **GNU Affero General Public License v3.0**
(ver `LICENSE`).

La elección no es arbitraria. El sistema depende de la biblioteca
[Ultralytics](https://github.com/ultralytics/ultralytics), licenciada bajo
AGPL-3.0, y distribuye archivos de pesos que declaran esa misma licencia en sus
propios metadatos. Verificado directamente sobre los archivos:

| Archivo | Licencia declarada en el propio archivo |
|---|---|
| `models/person_yolov8n.pt` | `AGPL-3.0 License (https://ultralytics.com/license)` |
| `models/person_yolov8n-pose.pt` | `AGPL-3.0 License (https://ultralytics.com/license)` |
| `models/weapon_yolov8s.pt` | `AGPL-3.0 (https://ultralytics.com/license)` |

> **Implicación para el despliegue.** Instalar este agente en el equipo de un
> negocio constituye distribución, lo que activa las obligaciones de la AGPL
> sobre el código del sistema enlazado. Si el proyecto evoluciona hacia un
> producto comercial cerrado, las alternativas son: exportar los modelos a ONNX
> y ejecutarlos con un motor de licencia permisiva sin empaquetar Ultralytics,
> adquirir una licencia comercial de Ultralytics, o migrar a un detector con
> licencia permisiva. La primera opción resuelve además el rendimiento.

---

## 2. Modelos incluidos

| Modelo | Origen | Observación |
|---|---|---|
| `person_yolov8n.pt` | Ultralytics, pesos oficiales COCO (30-12-2022) | Sin modificaciones |
| `person_yolov8n-pose.pt` | Ultralytics, pesos oficiales de pose | Sin modificaciones |
| `weapon_yolov8s.pt` | [WasifSohail5/Weapon-Detection-System](https://github.com/WasifSohail5/Weapon-Detection-System) (repositorio MIT) | Ajustado sobre un conjunto de datos privado **no disponible**. Su procedencia no es verificable |
| `violence_mc3_18.pth` | [MohamedSebaie/Fight_Detection_From_Surveillance_Cameras-PyTorch_Project](https://github.com/MohamedSebaie/Fight_Detection_From_Surveillance_Cameras-PyTorch_Project) (repositorio MIT) | Base Kinetics-400 ajustada sobre 300 clips de origen no documentado |

**Advertencia sobre los dos últimos.** Ambos repositorios se publican bajo
licencia MIT, pero ninguno documenta la licencia del material con el que se
entrenaron sus pesos. Esa incertidumbre se hereda. Antes de cualquier uso
comercial debería resolverse, preferentemente reentrenando sobre un conjunto de
datos de procedencia conocida.

---

## 3. Material de video

### Qué contiene

El repositorio incluye en `videos/entrada/` siete archivos de video, y en
`videos/salida/` y `eventos/` material derivado de ellos (videos anotados y
clips de evidencia).

Seis de esos archivos son **fragmentos de reportajes televisivos peruanos** que
difunden grabaciones de cámaras de seguridad de asaltos reales a comercios.
Las emisoras identificables en el material son Latina Noticias, Panamericana
Televisión (Buenos Días Perú) y 24 Horas. El séptimo es el video de muestra del
repositorio de detección de armas citado arriba.

### Por qué está incluido

El material se conserva en el repositorio para que el equipo del proyecto
disponga de la evidencia completa que sustenta las mediciones documentadas en
`README.md` y en el documento técnico entregable: sin los videos, los resultados
no son reproducibles ni verificables.

### Qué conviene tener presente

- **Son grabaciones de terceros.** No se dispone de licencia de las emisoras
  para redistribuirlas. Se incluyen con finalidad técnica y de investigación.
- **Muestran a personas identificables** —víctimas y presuntos autores de
  delitos reales— en establecimientos comerciales del Perú. Se trata de material
  ya difundido públicamente por dichas emisoras, pero su presencia en este
  repositorio amplía su alcance y permanencia.
- La documentación previa del proyecto (validación de detección de múltiples
  personas) estableció el criterio de **no redistribuir** este contenido y
  publicar únicamente las métricas obtenidas. La decisión de incluirlo aquí se
  toma de forma consciente y para uso del equipo.

### Si se desea publicar el proyecto sin el material

Basta con añadir estas líneas a `.gitignore` y eliminar los archivos del
historial:

```
videos/entrada/*
videos/salida/*
eventos/*
```

El código, los modelos y la documentación son autosuficientes; únicamente se
pierde la posibilidad de reproducir las mediciones sin aportar video propio.

---

## 4. Datos personales

El sistema **no realiza reconocimiento facial ni identificación de personas**.
El seguimiento asigna identificadores numéricos temporales, válidos únicamente
dentro de una misma cámara y sesión, que se descartan al terminar el análisis.

Los eventos generados no contienen datos personales: registran tipo de evento,
confianza, marca de tiempo, número aproximado de personas y la ruta del clip de
evidencia.

En un despliegue real, el tratamiento de imágenes de personas en
establecimientos comerciales queda sujeto a la normativa peruana de protección
de datos personales (Ley N.º 29733) y requiere la autorización del negocio, tal
como establece el alcance del proyecto.
