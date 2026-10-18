"""Clase ExtractorCaracteristicas (diagrama de clases, Fig. 1)."""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from .config import ConfigModelo
from .correo import PATRON_URL, CorreoElectronico

_PATRON_HTML = re.compile(r"<[^>]+>")
_PATRON_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PATRON_NUMERO = re.compile(r"\b\d+(?:[.,]\d+)*\b")
# Conserva letras (incluye acentos y ñ) y los tokens especiales.
_PATRON_NO_LETRA = re.compile(r"[^a-záéíóúüñ_\s]")
_PATRON_ESPACIOS = re.compile(r"\s+")


class ExtractorCaracteristicas:
    """Preprocesa el texto y lo vectoriza con TF-IDF.

    Se separa de `ModeloNaiveBayes` para que el preprocesamiento sea el
    mismo en entrenamiento y en producción (evita el desajuste
    entrenamiento/servicio).
    """

    def __init__(self, cfg: ConfigModelo = ConfigModelo()):
        self.cfg = cfg
        self.vectorizador_tfidf: TfidfVectorizer | None = None

    def preprocesar(self, correo: CorreoElectronico | str) -> str:
        """Normaliza: minúsculas, URLs -> 'urltoken', correos -> 'emailtoken',
        números -> 'numtoken'; elimina HTML y caracteres especiales.

        Decisión de diseño (D2): reemplazar las URLs por un token en lugar de
        borrarlas, porque la presencia de enlaces es una señal fuerte que el
        modelo de palabras no vería si se eliminan.
        """
        texto = correo.obtener_contenido() if isinstance(correo, CorreoElectronico) else str(correo)
        texto = texto.lower()
        texto = _PATRON_HTML.sub(" ", texto)
        texto = PATRON_URL.sub(" urltoken ", texto)
        texto = _PATRON_EMAIL.sub(" emailtoken ", texto)
        texto = _PATRON_NUMERO.sub(" numtoken ", texto)
        texto = _PATRON_NO_LETRA.sub(" ", texto)
        return _PATRON_ESPACIOS.sub(" ", texto).strip()

    def _crear_vectorizador(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            max_features=self.cfg.max_caracteristicas,
            ngram_range=(self.cfg.ngram_min, self.cfg.ngram_max),
            min_df=self.cfg.min_df,
            stop_words="english" if self.cfg.stopwords_ingles else None,
            sublinear_tf=True,
            preprocessor=None,  # el preprocesamiento ya se hizo en preprocesar()
            lowercase=False,
        )

    def ajustar(self, textos: list[str]) -> None:
        """Ajusta el vocabulario TF-IDF SOLO con el conjunto de entrenamiento."""
        self.vectorizador_tfidf = self._crear_vectorizador()
        self.vectorizador_tfidf.fit([self.preprocesar(t) for t in textos])

    def vectorizar(self, textos: list[str]):
        """Devuelve la matriz TF-IDF dispersa lista para el modelo."""
        if self.vectorizador_tfidf is None:
            raise RuntimeError("El extractor no está ajustado: llama primero a ajustar().")
        return self.vectorizador_tfidf.transform([self.preprocesar(t) for t in textos])

    @property
    def vocabulario(self) -> list[str]:
        if self.vectorizador_tfidf is None:
            raise RuntimeError("El extractor no está ajustado: llama primero a ajustar().")
        return list(self.vectorizador_tfidf.get_feature_names_out())
