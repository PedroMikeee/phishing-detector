"""Clase ModeloNaiveBayes (diagrama de clases, Fig. 1)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB

from .config import ETIQUETA_LEGITIMO, ETIQUETA_PHISHING, ConfigModelo


class ModeloNaiveBayes:
    """Naive Bayes Multinomial sobre vectores TF-IDF.

    Además de la probabilidad de phishing, expone los términos que más
    empujaron la decisión (explicabilidad básica, en línea con la brecha
    identificada en Trabajos Relacionados, sección II.C).

    Si se entrena con `calibrar=True`, las probabilidades que devuelve
    `predecir` salen de un calibrador sigmoide (Naive Bayes suele dar
    probabilidades extremas), mientras que la explicación de términos sigue
    usando el Naive Bayes sin calibrar, que es el que tiene pesos por término.
    """

    def __init__(self, cfg: ConfigModelo = ConfigModelo()):
        self.cfg = cfg
        self.clases_entrenadas: list[int] = []
        self.probabilidades: dict = {}   # probabilidades a priori de cada clase
        self._clasificador: MultinomialNB | None = None
        self._calibrador: CalibratedClassifierCV | None = None

    # ------------------------------------------------------------------ entrenamiento
    def entrenar(self, X_entrenamiento, y_entrenamiento, calibrar: bool = False) -> None:
        clf = MultinomialNB(alpha=self.cfg.alpha)
        clf.fit(X_entrenamiento, y_entrenamiento)
        self._clasificador = clf
        self.clases_entrenadas = [int(c) for c in clf.classes_]
        priors = np.exp(clf.class_log_prior_)
        self.probabilidades = {int(c): float(p) for c, p in zip(clf.classes_, priors)}

        self._calibrador = None
        if calibrar:
            cal = CalibratedClassifierCV(MultinomialNB(alpha=self.cfg.alpha), method="sigmoid", cv=5)
            cal.fit(X_entrenamiento, y_entrenamiento)
            self._calibrador = cal

    def _exigir_entrenado(self) -> MultinomialNB:
        if self._clasificador is None:
            raise RuntimeError("El modelo no está entrenado: llama primero a entrenar().")
        return self._clasificador

    # ------------------------------------------------------------------ predicción
    def predecir(self, X) -> np.ndarray:
        """Probabilidad de phishing (clase 1) para cada fila de X, en [0, 1]."""
        clf = self._exigir_entrenado()
        modelo = self._calibrador if self._calibrador is not None else clf
        columna = list(modelo.classes_).index(ETIQUETA_PHISHING)
        return modelo.predict_proba(X)[:, columna]

    # ------------------------------------------------------------------ explicación
    def _diferencia_log_prob(self) -> np.ndarray:
        """log P(t | malicioso) - log P(t | legítimo) para cada término."""
        clf = self._exigir_entrenado()
        clases = list(clf.classes_)
        return clf.feature_log_prob_[clases.index(ETIQUETA_PHISHING)] - \
            clf.feature_log_prob_[clases.index(ETIQUETA_LEGITIMO)]

    def terminos_influyentes(self, x_fila, vocabulario: list[str], n: int = 5) -> list[tuple[str, float]]:
        """Los n términos presentes en el correo con mayor aporte hacia phishing.

        Aporte de un término = valor TF-IDF * (log P(t|phishing) - log P(t|legítimo)).
        Solo se devuelven aportes positivos (los que empujan hacia malicioso).
        """
        diff = self._diferencia_log_prob()
        x = csr_matrix(x_fila)
        indices, valores = x.indices, x.data
        aportes = valores * diff[indices]
        orden = np.argsort(aportes)[::-1]
        return [(vocabulario[indices[i]], float(aportes[i])) for i in orden[:n] if aportes[i] > 0]

    def terminos_mas_discriminantes(self, vocabulario: list[str], n: int = 15):
        """Términos del vocabulario que más inclinan hacia cada clase.

        Sirve para auditar el modelo: si aparecen términos que no tienen que ver
        con el contenido (fechas, nombres de listas), el modelo está aprendiendo
        rasgos de la fuente y no señales reales.
        Devuelve (hacia_malicioso, hacia_legitimo), cada una como [(término, peso)].
        """
        diff = self._diferencia_log_prob()
        orden = np.argsort(diff)
        hacia_legitimo = [(vocabulario[i], float(diff[i])) for i in orden[:n]]
        hacia_malicioso = [(vocabulario[i], float(diff[i])) for i in orden[::-1][:n]]
        return hacia_malicioso, hacia_legitimo

    # ------------------------------------------------------------------ persistencia
    def guardar(self, ruta: Path) -> None:
        self._exigir_entrenado()
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ruta)

    @classmethod
    def cargar(cls, ruta: Path) -> "ModeloNaiveBayes":
        modelo = joblib.load(ruta)
        if not isinstance(modelo, cls):
            raise TypeError(f"{ruta} no contiene un ModeloNaiveBayes")
        return modelo
