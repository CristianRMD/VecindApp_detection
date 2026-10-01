"""Evento segun el contrato de RESUMEN seccion 13."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

TIPOS = ("suspicious_weapon", "physical_violence", "possible_robbery", "possible_multiple_robbery")

# Nombre legible del tipo, para el NOMBRE DE ARCHIVO. El campo event_type del
# JSON conserva siempre el valor en ingles del contrato de RESUMEN seccion 13.
SLUG = {
    "suspicious_weapon": "arma-sospechosa",
    "physical_violence": "violencia-fisica",
    "possible_robbery": "posible-atraco",
    "possible_multiple_robbery": "posible-atraco-multiple",
}

# Prioridad para decidir si un evento nuevo ESCALA a otro ya emitido
PRIORIDAD = {
    "suspicious_weapon": 1,
    "physical_violence": 1,
    "possible_robbery": 2,
    "possible_multiple_robbery": 3,
}


@dataclass
class Evento:
    event_type: str
    severity: str
    confidence: float
    detected_at: str
    t_video: float                       # segundos dentro del archivo analizado
    people_count: int
    atacantes: int
    detections: list[dict] = field(default_factory=list)
    zone: str | None = None
    business_id: str = "business-001"
    camera_id: str = "cam-01"
    origen: str = "video"                # archivo del que salio, para el nombre
    event_id: str = ""
    evidence_url: str | None = None
    review_status: str = "pending_review"

    def __post_init__(self):
        if not self.event_id:
            # Unico (marca de tiempo + sufijo aleatorio) pero tambien legible:
            # se ve de que camara y de cuando es sin abrir el archivo.
            marca = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            self.event_id = f"evt-{self.camera_id}-{marca}-{uuid.uuid4().hex[:3]}"

    @property
    def base(self) -> str:
        """
        Nombre de archivo legible: ordena por video y, dentro de cada video,
        cronologicamente.
            Video5__00m07s__posible-atraco-multiple__c78
        Es DETERMINISTA a proposito: volver a analizar el mismo video reemplaza
        sus eventos en vez de acumular duplicados.
        """
        minutos, segundos = divmod(int(self.t_video), 60)
        tipo = SLUG.get(self.event_type, self.event_type)
        return f"{self.origen}__{minutos:02d}m{segundos:02d}s__{tipo}__c{round(self.confidence * 100):02d}"

    @staticmethod
    def ahora_iso() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def a_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "business_id": self.business_id,
            "camera_id": self.camera_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "confidence": round(self.confidence, 4),
            "detected_at": self.detected_at,
            "zone": self.zone,
            "people_count": self.people_count,
            "detections": self.detections,
            "evidence_url": self.evidence_url,
            "review_status": self.review_status,
            # --- extras locales, no forman parte del contrato de la API ---
            "_t_video_seg": round(self.t_video, 2),
            "_atacantes": self.atacantes,
        }

    def guardar(self, carpeta: Path) -> Path:
        carpeta.mkdir(parents=True, exist_ok=True)
        destino = carpeta / f"{self.base}.json"
        destino.write_text(
            json.dumps(self.a_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return destino
