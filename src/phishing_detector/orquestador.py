"""Clase AgenteOrquestador (diagrama de clases, Fig. 1).

Es el único punto de entrada del sistema: la interfaz (Streamlit) y los
scripts solo hablan con este agente.
"""
from __future__ import annotations

from .alerta import Alerta, ModuloAlerta, NivelRiesgo
from .caracteristicas import ExtractorCaracteristicas
from .config import ConfigOrquestador
from .correo import CorreoElectronico
from .modelo import ModeloNaiveBayes


class AgenteOrquestador:
    """Coordina: correo -> extractor -> modelo -> nivel de riesgo -> alerta."""

    def __init__(self, extractor: ExtractorCaracteristicas, modelo: ModeloNaiveBayes,
                 modulo_alerta: ModuloAlerta | None = None,
                 cfg: ConfigOrquestador = ConfigOrquestador()):
        self.extractor = extractor
        self.modelo = modelo
        self.modulo_alerta = modulo_alerta or ModuloAlerta()
        self.cfg = cfg
        self.umbral_riesgo = cfg.umbral_alto
        self.nivel_alerta: NivelRiesgo | None = None

    def recibir_correo(self, correo: CorreoElectronico) -> Alerta:
        """Flujo completo; devuelve la alerta lista para mostrar."""
        raise NotImplementedError("Fase de implementación")

    def evaluar_riesgo(self, probabilidad: float) -> NivelRiesgo:
        """Traduce probabilidad a nivel con los umbrales de ConfigOrquestador.

        p < umbral_medio -> BAJO; umbral_medio <= p < umbral_alto -> MEDIO;
        p >= umbral_alto -> ALTO.
        """
        raise NotImplementedError("Fase de implementación")

    def generar_recomendacion(self, nivel: NivelRiesgo) -> list[str]:
        raise NotImplementedError("Fase de implementación")
