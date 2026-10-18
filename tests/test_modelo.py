"""Pruebas de B.4/B.5 con datos sintéticos separables."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishing_detector.caracteristicas import ExtractorCaracteristicas  # noqa: E402
from phishing_detector.config import ConfigModelo  # noqa: E402
from phishing_detector.evaluacion import evaluar_probabilidades, nivel_de_riesgo  # noqa: E402
from phishing_detector.modelo import ModeloNaiveBayes  # noqa: E402

LEGIT = ["meeting", "project", "lunch", "report", "schedule", "thanks", "review", "team", "draft", "budget"]
MALO = ["verify", "account", "urgent", "click", "winner", "bank", "password", "prize", "login", "offer"]


def _corpus(n=150, semilla=0):
    rng = np.random.default_rng(semilla)
    textos, y = [], []
    for _ in range(n):
        textos.append(" ".join(rng.choice(LEGIT, 12)) + " " + rng.choice(MALO))
        y.append(0)
        textos.append(" ".join(rng.choice(MALO, 12)) + " " + rng.choice(LEGIT))
        y.append(1)
    return textos, np.array(y)


@pytest.fixture(scope="module")
def entrenado():
    cfg = ConfigModelo(min_df=1)
    textos, y = _corpus()
    ex = ExtractorCaracteristicas(cfg)
    ex.ajustar(textos)
    modelo = ModeloNaiveBayes(cfg)
    modelo.entrenar(ex.vectorizar(textos), y)
    return ex, modelo


def test_predecir_devuelve_probabilidades_validas(entrenado):
    ex, modelo = entrenado
    textos, y = _corpus(semilla=1)
    p = modelo.predecir(ex.vectorizar(textos))
    assert p.shape == (len(textos),)
    assert ((p >= 0) & (p <= 1)).all()
    m = evaluar_probabilidades(y, p)
    assert m["exactitud"] > 0.95 and m["auc_roc"] > 0.95


def test_no_entrenado_falla():
    with pytest.raises(RuntimeError):
        ModeloNaiveBayes().predecir(None)


def test_terminos_influyentes_apuntan_a_malicioso(entrenado):
    ex, modelo = entrenado
    x = ex.vectorizar(["urgent verify your account password now"])
    terminos = modelo.terminos_influyentes(x, ex.vocabulario, n=3)
    assert 1 <= len(terminos) <= 3
    # el vocabulario incluye unigramas y bigramas: cada palabra debe ser "maliciosa"
    assert all(set(t.split()) <= set(MALO) for t, _ in terminos)
    assert all(aporte > 0 for _, aporte in terminos)


def test_terminos_discriminantes_separan_clases(entrenado):
    ex, modelo = entrenado
    mal, leg = modelo.terminos_mas_discriminantes(ex.vocabulario, n=5)
    assert all(set(t.split()) <= set(MALO) for t, _ in mal)
    assert all(set(t.split()) <= set(LEGIT) for t, _ in leg)


def test_guardar_y_cargar_dan_las_mismas_predicciones(entrenado, tmp_path):
    ex, modelo = entrenado
    ruta = tmp_path / "m.joblib"
    modelo.guardar(ruta)
    otro = ModeloNaiveBayes.cargar(ruta)
    X = ex.vectorizar(_corpus(20, semilla=2)[0])
    assert np.allclose(modelo.predecir(X), otro.predecir(X))


def test_calibrado_sigue_dando_probabilidades_validas():
    cfg = ConfigModelo(min_df=1)
    textos, y = _corpus()
    ex = ExtractorCaracteristicas(cfg)
    ex.ajustar(textos)
    modelo = ModeloNaiveBayes(cfg)
    modelo.entrenar(ex.vectorizar(textos), y, calibrar=True)
    p = modelo.predecir(ex.vectorizar(textos))
    assert ((p >= 0) & (p <= 1)).all()
    # el explicador sigue funcionando con calibrador activo
    assert modelo.terminos_influyentes(ex.vectorizar(["verify login"]), ex.vocabulario, 2)


def test_niveles_de_riesgo_y_reparto():
    assert nivel_de_riesgo(0.10) == "bajo"
    assert nivel_de_riesgo(0.35) == "medio"
    assert nivel_de_riesgo(0.69) == "medio"
    assert nivel_de_riesgo(0.70) == "alto"
    m = evaluar_probabilidades([0, 0, 1, 1], [0.1, 0.5, 0.6, 0.9])
    assert m["niveles_riesgo_pct"]["legitimo"] == {"bajo": 50.0, "medio": 50.0, "alto": 0.0}
    assert m["niveles_riesgo_pct"]["malicioso"] == {"bajo": 0.0, "medio": 50.0, "alto": 50.0}
    assert m["matriz_confusion"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}


def test_r2_y_mse_de_probabilidades_perfectas():
    m = evaluar_probabilidades([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0])
    assert m["mse_probabilidad"] == 0.0 and m["r2_probabilidad"] == 1.0
