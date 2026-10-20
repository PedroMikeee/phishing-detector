"""Clase ModuloAlerta (diagrama de clases, Fig. 1).

Objetivo específico 3 del anteproyecto (Mitigación): según el nivel de riesgo,
informar al usuario de forma clara y recomendarle acciones de precaución. Los
textos de cada nivel son los de docs/DISENO.md, sección 6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .idioma import INDETERMINADO


class NivelRiesgo(str, Enum):
    BAJO = "bajo"
    MEDIO = "medio"
    ALTO = "alto"

    @property
    def orden(self) -> int:
        return _ORDEN[self]


_ORDEN = {NivelRiesgo.BAJO: 0, NivelRiesgo.MEDIO: 1, NivelRiesgo.ALTO: 2}


def nivel_maximo(*niveles: NivelRiesgo) -> NivelRiesgo:
    """El más grave de los niveles dados."""
    return max(niveles, key=lambda n: n.orden)


MENSAJES = {
    NivelRiesgo.ALTO: "Riesgo ALTO: este correo muy probablemente es phishing o fraude.",
    NivelRiesgo.MEDIO: "Riesgo MEDIO: hay señales mixtas; verifica antes de actuar.",
    NivelRiesgo.BAJO: "Riesgo BAJO: no se encontraron señales fuertes de phishing.",
}

# Cuando no se pudo usar el modelo (idioma no compatible) un nivel bajo no
# significa "seguro": solo que las reglas no encontraron nada.
MENSAJE_BAJO_SIN_MODELO = ("Riesgo BAJO según las reglas, pero el contenido del correo no se pudo "
                           "analizar con el modelo.")

ACCIONES = {
    NivelRiesgo.ALTO: [
        "No hagas clic en ningún enlace.",
        "No descargues adjuntos ni respondas el correo.",
        "Verifica al remitente por otro canal (por ejemplo, escribe tú mismo la dirección oficial del sitio).",
        "Reporta el correo y elimínalo.",
    ],
    NivelRiesgo.MEDIO: [
        "No hagas clic todavía.",
        "Comprueba el dominio del remitente y hacia dónde apuntan los enlaces.",
        "Ante la duda, trátalo como riesgo alto.",
    ],
    NivelRiesgo.BAJO: [
        "Mantén la precaución habitual con enlaces y adjuntos inesperados.",
    ],
}

ACCION_BAJO_SIN_MODELO = "Ante la duda, verifica el correo por otro canal."


@dataclass
class Alerta:
    """Resultado que ve el usuario final."""

    nivel: NivelRiesgo                  # nivel final (modelo + reglas)
    probabilidad: float | None          # P(phishing) del modelo; None si no se usó el modelo
    mensaje: str                        # texto principal de la alerta
    acciones_sugeridas: list[str] = field(default_factory=list)
    terminos_sospechosos: list[tuple[str, float]] = field(default_factory=list)
    enlaces_detectados: list[str] = field(default_factory=list)
    # Añadidos con las decisiones D7 y D8 (docs/DISENO.md):
    motivos: list[str] = field(default_factory=list)        # explicaciones de las reglas
    advertencias: list[str] = field(default_factory=list)   # límites del análisis (idioma, etc.)
    idioma: str = INDETERMINADO
    nivel_modelo: NivelRiesgo | None = None                 # nivel que daba solo el modelo
    puntos_reglas: int = 0


class ModuloAlerta:
    """Construye y presenta la alerta (objetivo específico 3: Mitigación)."""

    def __init__(self):
        self.mensaje: str = ""
        self.acciones_sugeridas: list[str] = []
        self.registro: list[Alerta] = []   # historial para registrar_evento

    def construir_alerta(self, nivel: NivelRiesgo, probabilidad: float | None,
                         terminos: list[tuple[str, float]],
                         enlaces: list[str], *,
                         motivos: list[str] | None = None,
                         advertencias: list[str] | None = None,
                         idioma: str = INDETERMINADO,
                         nivel_modelo: NivelRiesgo | None = None,
                         puntos_reglas: int = 0) -> Alerta:
        """Mensaje + acciones según el nivel (textos de docs/DISENO.md §6)."""
        nivel = NivelRiesgo(nivel)
        sin_modelo = probabilidad is None
        mensaje = MENSAJE_BAJO_SIN_MODELO if (sin_modelo and nivel is NivelRiesgo.BAJO) else MENSAJES[nivel]
        acciones = list(ACCIONES[nivel])
        if sin_modelo and nivel is NivelRiesgo.BAJO:
            acciones.append(ACCION_BAJO_SIN_MODELO)
        self.mensaje, self.acciones_sugeridas = mensaje, acciones
        return Alerta(
            nivel=nivel, probabilidad=probabilidad, mensaje=mensaje, acciones_sugeridas=acciones,
            terminos_sospechosos=list(terminos), enlaces_detectados=list(enlaces),
            motivos=list(motivos or []), advertencias=list(advertencias or []),
            idioma=idioma, nivel_modelo=nivel_modelo, puntos_reglas=puntos_reglas,
        )

    def mostrar_alerta(self, alerta: Alerta) -> str:
        """Texto plano de la alerta (la interfaz Streamlit la renderiza con color)."""
        cabecera = f"{alerta.nivel.value.upper()}"
        if alerta.probabilidad is not None:
            cabecera += f" — probabilidad del modelo: {_porcentaje(alerta.probabilidad)}"
        lineas = [cabecera, alerta.mensaje]

        if alerta.motivos:
            lineas += ["", "Por qué:"] + [f"  - {m}" for m in alerta.motivos]
        # Las palabras solo explican cuando el modelo ve riesgo; en un nivel bajo confundirían.
        if alerta.terminos_sospechosos and alerta.nivel_modelo not in (None, NivelRiesgo.BAJO):
            palabras = ", ".join(t for t, _ in alerta.terminos_sospechosos)
            lineas += ["", f"Palabras que más pesaron en el modelo: {palabras}"]
        if alerta.enlaces_detectados:
            lineas += ["", "Enlaces detectados (no los abras):"] + [f"  - {e}" for e in alerta.enlaces_detectados]
        lineas += ["", "Qué hacer:"] + [f"  {i}. {a}" for i, a in enumerate(alerta.acciones_sugeridas, 1)]
        if alerta.advertencias:
            lineas += ["", "Avisos:"] + [f"  ! {a}" for a in alerta.advertencias]
        return "\n".join(lineas)

    def registrar_evento(self, alerta: Alerta) -> None:
        self.registro.append(alerta)


def _porcentaje(p: float) -> str:
    return f"{p * 100:.1f} %".replace(".", ",")
