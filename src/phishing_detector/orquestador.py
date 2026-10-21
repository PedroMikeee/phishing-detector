"""Clase AgenteOrquestador (diagrama de clases, Fig. 1).

Es el único punto de entrada del sistema: la interfaz (Streamlit) y los
scripts solo hablan con este agente.

Flujo de `recibir_correo` (docs/DISENO.md, secciones 4 y 5):

    correo -> idioma -> reglas (enlaces, remitente, texto)
                     -> [si el idioma lo permite] extractor -> modelo -> P(phishing)
           -> nivel del modelo + nivel de las reglas -> el más grave -> alerta

Las reglas solo pueden subir el nivel que da el modelo, nunca bajarlo.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from . import idioma as _idioma
from .alerta import ACCIONES, Alerta, ModuloAlerta, NivelRiesgo, nivel_maximo
from .caracteristicas import ExtractorCaracteristicas
from .config import ETIQUETA_LEGITIMO, ETIQUETA_PHISHING, RUTA_MODELO_ENTRENADO, ConfigOrquestador
from .correo import PATRON_URL, CorreoElectronico
from .evaluacion import nivel_de_riesgo
from .modelo import ModeloNaiveBayes
from .reglas import evaluar_reglas, total_puntos


class AgenteOrquestador:
    """Coordina: correo -> extractor -> modelo -> reglas -> nivel de riesgo -> alerta."""

    def __init__(self, extractor: ExtractorCaracteristicas, modelo: ModeloNaiveBayes,
                 modulo_alerta: ModuloAlerta | None = None,
                 cfg: ConfigOrquestador = ConfigOrquestador()):
        self.extractor = extractor
        self.modelo = modelo
        self.modulo_alerta = modulo_alerta or ModuloAlerta()
        self.cfg = cfg
        self.umbral_riesgo = cfg.umbral_alto
        self.nivel_alerta: NivelRiesgo | None = None
        self._vocabulario: list[str] | None = None   # se calcula una vez (20 000 términos)
        self.metricas: dict | None = None            # métricas del entrenamiento (si se cargó de archivo)

    @classmethod
    def desde_archivo(cls, ruta: Path = RUTA_MODELO_ENTRENADO,
                      cfg: ConfigOrquestador = ConfigOrquestador()) -> "AgenteOrquestador":
        """Carga el extractor y el modelo que guarda `scripts/entrenar.py`."""
        ruta = Path(ruta)
        if not ruta.exists():
            raise FileNotFoundError(f"No hay modelo entrenado en {ruta}. Ejecuta: python scripts/entrenar.py")
        guardado = joblib.load(ruta)
        agente = cls(guardado["extractor"], guardado["modelo"], cfg=cfg)
        agente.metricas = guardado.get("metricas")   # métricas del entrenamiento, para la interfaz
        return agente

    def guardar(self, ruta: Path = RUTA_MODELO_ENTRENADO) -> None:
        """Guarda extractor, modelo y métricas en el mismo formato que `scripts/entrenar.py`."""
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"extractor": self.extractor, "modelo": self.modelo, "config": self.modelo.cfg,
                     "metricas": self.metricas}, ruta)

    # ------------------------------------------------------------------ actualización (D9)
    def aprender(self, correos: list[CorreoElectronico | str], etiquetas: list[int],
                 descartar_calibracion: bool = False) -> int:
        """Incorpora correos nuevos etiquetados (0 = legítimo, 1 = malicioso) al modelo.

        Devuelve cuántos ejemplos se incorporaron. Ver `ModeloNaiveBayes.actualizar`
        para los límites (vocabulario fijo, calibrador).
        """
        if len(correos) != len(etiquetas):
            raise ValueError("Debe haber una etiqueta por cada correo.")
        if not correos:
            return 0
        if not set(etiquetas) <= {ETIQUETA_LEGITIMO, ETIQUETA_PHISHING}:
            raise ValueError("Las etiquetas deben ser 0 (legítimo) o 1 (malicioso).")
        textos = [c.obtener_contenido() if isinstance(c, CorreoElectronico) else str(c) for c in correos]
        if any(not t.strip() for t in textos):
            raise ValueError("Hay correos vacíos entre los ejemplos nuevos.")
        X = self.extractor.vectorizar(textos)
        self.modelo.actualizar(X, np.asarray(etiquetas, dtype=int), descartar_calibracion=descartar_calibracion)
        return len(textos)

    # ------------------------------------------------------------------ flujo principal
    def recibir_correo(self, correo: CorreoElectronico | str) -> Alerta:
        """Flujo completo; devuelve la alerta lista para mostrar.

        Acepta un `CorreoElectronico` o el texto pegado por el usuario (con
        cabeceras From:/Subject: opcionales).
        """
        if isinstance(correo, str):
            correo = CorreoElectronico.desde_texto_plano(correo)
        contenido = correo.obtener_contenido()
        if not contenido:
            raise ValueError("El correo está vacío: no hay nada que analizar.")

        idioma = _idioma.detectar_idioma(contenido)
        hallazgos = evaluar_reglas(correo)
        puntos = total_puntos(hallazgos)
        nivel_reglas = self._nivel_por_puntos(puntos)
        advertencias: list[str] = []

        if not self.cfg.verificar_idioma or _idioma.modelo_aplicable(idioma):
            x = self.extractor.vectorizar([contenido])
            probabilidad = float(self.modelo.predecir(x)[0])
            terminos = self.modelo.terminos_influyentes(
                x, self._obtener_vocabulario(), n=self.cfg.n_terminos_explicacion)
            nivel_modelo = self.evaluar_riesgo(probabilidad)
        else:
            probabilidad, terminos, nivel_modelo = None, [], NivelRiesgo.BAJO
            advertencias.append(
                f"El modelo solo se entrenó con correos en inglés y este parece estar en "
                f"{_idioma.NOMBRES[idioma]}: no se usó su probabilidad. El nivel sale solo de las "
                "reglas sobre enlaces, remitente y frases típicas.")

        nivel = nivel_maximo(nivel_modelo, nivel_reglas)
        enlaces = correo.enlaces or self._enlaces_del_texto(correo)
        alerta = self.modulo_alerta.construir_alerta(
            nivel, probabilidad, terminos, enlaces,
            motivos=[h.descripcion for h in hallazgos], advertencias=advertencias,
            idioma=idioma, nivel_modelo=nivel_modelo, puntos_reglas=puntos)
        self.nivel_alerta = nivel
        self.modulo_alerta.registrar_evento(alerta)
        return alerta

    def evaluar_riesgo(self, probabilidad: float) -> NivelRiesgo:
        """Traduce probabilidad a nivel con los umbrales de ConfigOrquestador.

        p < umbral_medio -> BAJO; umbral_medio <= p < umbral_alto -> MEDIO;
        p >= umbral_alto -> ALTO. Usa la misma regla que `evaluacion.nivel_de_riesgo`,
        con la que se calcularon las métricas, para que no se desvíen.
        """
        return NivelRiesgo(nivel_de_riesgo(probabilidad, self.cfg))

    def generar_recomendacion(self, nivel: NivelRiesgo) -> list[str]:
        return list(ACCIONES[NivelRiesgo(nivel)])

    # ------------------------------------------------------------------ auxiliares
    def _nivel_por_puntos(self, puntos: int) -> NivelRiesgo:
        if puntos >= self.cfg.puntos_alto:
            return NivelRiesgo.ALTO
        if puntos >= self.cfg.puntos_medio:
            return NivelRiesgo.MEDIO
        return NivelRiesgo.BAJO

    def _obtener_vocabulario(self) -> list[str]:
        if self._vocabulario is None:
            self._vocabulario = self.extractor.vocabulario
        return self._vocabulario

    @staticmethod
    def _enlaces_del_texto(correo: CorreoElectronico) -> list[str]:
        return [u.rstrip(".,;:") for u in PATRON_URL.findall(correo.obtener_contenido())]
