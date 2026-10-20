"""Configuración central del mecanismo.

Todos los parámetros que se van a ajustar durante la implementación y la
evaluación viven aquí, para que el diseño sea trazable (CRISP-DM, B.4.2).
"""
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
RUTA_DATOS_CRUDOS = RAIZ / "data" / "raw"
RUTA_DATOS_PROCESADOS = RAIZ / "data" / "processed"
RUTA_MODELOS = RAIZ / "models"
RUTA_MODELO_ENTRENADO = RUTA_MODELOS / "modelo_phishing.joblib"

# Etiquetas internas del sistema (0 = legítimo, 1 = phishing).
ETIQUETA_LEGITIMO = 0
ETIQUETA_PHISHING = 1


@dataclass(frozen=True)
class ConfigModelo:
    """Hiperparámetros del vectorizador TF-IDF y del Naive Bayes (B.4.2)."""

    max_caracteristicas: int = 20_000   # tamaño del vocabulario TF-IDF
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2                     # ignora términos casi inexistentes
    stopwords_ingles: bool = True       # el corpus actual está en inglés
    alpha: float = 0.5                  # suavizado de Laplace/Lidstone
    proporcion_prueba: float = 0.2      # 80/20 estratificado
    semilla: int = 42


@dataclass(frozen=True)
class ConfigOrquestador:
    """Umbrales que traducen probabilidad en nivel de riesgo.

    Valores iniciales: se calibran en la fase de evaluación con el conjunto
    de validación (ver docs/DISENO.md, sección 5).
    """

    umbral_medio: float = 0.35   # p < 0.35  -> bajo
    umbral_alto: float = 0.70    # p >= 0.70 -> alto
    n_terminos_explicacion: int = 5
    # Reglas complementarias (reglas.py): cada hallazgo suma puntos (1 débil,
    # 2 media, 3 fuerte) y el total fija un nivel mínimo, independiente del modelo.
    puntos_medio: int = 2        # >= 2 puntos -> al menos medio
    puntos_alto: int = 4         # >= 4 puntos -> alto


# Nombres de columnas aceptados en los CSV de entrada (ver data.py).
# Se comparan en minúsculas y sin espacios sobrantes.
COLUMNAS_ASUNTO = ("subject", "asunto")
COLUMNAS_CUERPO = ("body", "text", "email_text", "email text", "text_combined", "message", "content")
COLUMNAS_ETIQUETA = ("label", "type", "email_type", "email type", "class", "target")
COLUMNAS_REMITENTE = ("sender", "from", "remitente")
COLUMNAS_URLS = ("urls", "has_urls", "tiene_urls")

# Mapeo de etiquetas externas a las internas. Cualquier valor que no esté
# aquí hace fallar la carga: es preferible parar a entrenar con etiquetas
# mal interpretadas. Las claves van en minúsculas.
MAPA_ETIQUETAS = {
    "1": ETIQUETA_PHISHING, "phishing": ETIQUETA_PHISHING,
    "phishing email": ETIQUETA_PHISHING, "spam": ETIQUETA_PHISHING,
    "fraud": ETIQUETA_PHISHING,
    "0": ETIQUETA_LEGITIMO, "ham": ETIQUETA_LEGITIMO,
    "safe": ETIQUETA_LEGITIMO, "safe email": ETIQUETA_LEGITIMO,
    "legitimate": ETIQUETA_LEGITIMO, "legit": ETIQUETA_LEGITIMO,
}
