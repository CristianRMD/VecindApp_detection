"""Carga de config.yaml con acceso por punto y rutas resueltas contra la raiz."""

from __future__ import annotations

from pathlib import Path

import yaml


class Cfg(dict):
    """dict con acceso por atributo, recursivo. cfg.personas.activo.fps"""

    def __getattr__(self, clave):
        try:
            valor = self[clave]
        except KeyError as exc:
            raise AttributeError(clave) from exc
        return Cfg(valor) if isinstance(valor, dict) else valor

    def __setattr__(self, clave, valor):
        self[clave] = valor


class Config(Cfg):
    raiz: Path

    def ruta(self, relativa: str) -> Path:
        """Resuelve una ruta del yaml contra la carpeta del proyecto."""
        p = Path(relativa)
        return p if p.is_absolute() else self["raiz"] / p


def cargar(ruta_yaml: str | Path) -> Config:
    ruta_yaml = Path(ruta_yaml)
    datos = yaml.safe_load(ruta_yaml.read_text(encoding="utf-8"))
    cfg = Config(datos)
    cfg["raiz"] = ruta_yaml.parent
    return cfg
