"""Clase ModuloAlerta (diagrama de clases, Fig. 1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class NivelRiesgo(str, Enum):
    BAJO = "bajo"
    MEDIO = "medio"
    ALTO = "alto"


@dataclass
class Alerta:
    """Resultado que ve el usuario final."""

    nivel: NivelRiesgo
    probabilidad: float                 # P(phishing) del modelo, en [0, 1]
    mensaje: str                        # texto principal de la alerta
    acciones_sugeridas: list[str] = field(default_factory=list)
    terminos_sospechosos: list[tuple[str, float]] = field(default_factory=list)
    enlaces_detectados: list[str] = field(default_factory=list)


class ModuloAlerta:
    """Construye y presenta la alerta (objetivo específico 3: Mitigación)."""

    def __init__(self):
        self.mensaje: str = ""
        self.acciones_sugeridas: list[str] = []
        self.registro: list[Alerta] = []   # historial para registrar_evento

    def construir_alerta(self, nivel: NivelRiesgo, probabilidad: float,
                         terminos: list[tuple[str, float]],
                         enlaces: list[str]) -> Alerta:
        """Mensaje + acciones según el nivel (textos definidos en docs/DISENO.md §6)."""
        raise NotImplementedError("Fase de implementación")

    def mostrar_alerta(self, alerta: Alerta) -> str:
        """Texto plano de la alerta (la interfaz Streamlit la renderiza con color)."""
        raise NotImplementedError("Fase de implementación")

    def registrar_evento(self, alerta: Alerta) -> None:
        self.registro.append(alerta)
