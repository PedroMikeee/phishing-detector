"""Detección de idioma por palabras vacías (decisión D7 de docs/DISENO.md).

El modelo Naive Bayes solo se entrenó con correos en inglés. Cuando el texto
está en otro idioma casi ninguna palabra está en su vocabulario y la
probabilidad que devuelve no es fiable (un pedido legítimo en español dio
P = 0,93). Esta heurística, sin dependencias externas, permite avisarlo.

No pretende ser un detector de idioma general: solo distingue inglés de
español y "otro", suficiente para decidir si se confía en el modelo.
"""
from __future__ import annotations

import re

from .correo import PATRON_URL

_PATRON_PALABRA = re.compile(r"[^\W\d_]+")
_PATRON_EMAIL = re.compile(r"\S+@\S+")

# Palabras vacías frecuentes. Se excluyen las que existen en ambos idiomas
# ("a", "no", "me", "he", "son", "pin") para que no sumen a ninguno.
_INGLES = frozenset("""
the and to of in is you that it for on with as are this be your have from or at by we will not can
an our if please has was i my but all they there their what which would been were more when who do
does how so about here up out just he she his her its them these those than then also any
""".split())

_ESPANOL = frozenset("""
de la que el en y los se del las un por con una su para es al lo como más pero sus ya este esta
entre muy sin sobre también hay nos usted ustedes hola gracias fue ha han desde puede tiene mi tu
te le les estimado estimada
""".split())

# Fracción mínima de palabras reconocidas para afirmar un idioma.
_UMBRAL_FRACCION = 0.12
# Con menos palabras que esto no hay base para decidir.
_MIN_PALABRAS = 4
# Con tantas palabras o más y sin rasgos de inglés ni de español: "otro".
_MIN_PALABRAS_OTRO = 8

INGLES = "en"
ESPANOL = "es"
OTRO = "otro"
INDETERMINADO = "indeterminado"

NOMBRES = {INGLES: "inglés", ESPANOL: "español", OTRO: "un idioma no reconocido",
           INDETERMINADO: "indeterminado"}


def detectar_idioma(texto: str) -> str:
    """Devuelve "en", "es", "otro" o "indeterminado".

    Se ignoran enlaces y direcciones de correo. "indeterminado" aparece en
    textos demasiado cortos o con pocas palabras reconocibles.
    """
    limpio = _PATRON_EMAIL.sub(" ", PATRON_URL.sub(" ", texto.lower()))
    palabras = _PATRON_PALABRA.findall(limpio)
    n = len(palabras)
    if n < _MIN_PALABRAS:
        return INDETERMINADO
    f_en = sum(p in _INGLES for p in palabras) / n
    f_es = sum(p in _ESPANOL for p in palabras) / n
    if f_en >= _UMBRAL_FRACCION and f_en > f_es:
        return INGLES
    if f_es >= _UMBRAL_FRACCION and f_es > f_en:
        return ESPANOL
    return OTRO if n >= _MIN_PALABRAS_OTRO else INDETERMINADO


def modelo_aplicable(idioma: str) -> bool:
    """El modelo solo es fiable en inglés. En textos muy cortos
    ("indeterminado") se usa de todos modos: no hay evidencia de otro idioma."""
    return idioma in (INGLES, INDETERMINADO)
