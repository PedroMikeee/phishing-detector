"""Pruebas de la actualización del modelo con ejemplos nuevos (D9)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishing_detector.caracteristicas import ExtractorCaracteristicas  # noqa: E402
from phishing_detector.config import ConfigModelo  # noqa: E402
from phishing_detector.modelo import ModeloNaiveBayes  # noqa: E402
from phishing_detector.orquestador import AgenteOrquestador  # noqa: E402

LEGIT = ["meeting", "project", "lunch", "report", "schedule", "thanks", "review", "team", "draft", "budget"]
MALO = ["verify", "account", "urgent", "click", "winner", "bank", "password", "prize", "login", "offer"]
# Texto con palabras del vocabulario que el modelo asocia con correo legítimo.
TEXTO = "the team will review the lunch meeting and the project report"


def _corpus(n=150, semilla=0):
    rng = np.random.default_rng(semilla)
    textos, y = [], []
    for _ in range(n):
        textos.append(" ".join(rng.choice(LEGIT, 12)) + " " + rng.choice(MALO))
        y.append(0)
        textos.append(" ".join(rng.choice(MALO, 12)) + " " + rng.choice(LEGIT))
        y.append(1)
    return textos, np.array(y)


def _agente(calibrar=False):
    cfg = ConfigModelo(min_df=1)
    textos, y = _corpus()
    ex = ExtractorCaracteristicas(cfg)
    ex.ajustar(textos)
    modelo = ModeloNaiveBayes(cfg)
    modelo.entrenar(ex.vectorizar(textos), y, calibrar=calibrar)
    return AgenteOrquestador(ex, modelo)


def test_aprender_ejemplos_nuevos_cambia_la_probabilidad():
    agente = _agente()
    antes = agente.recibir_correo(TEXTO).probabilidad
    assert antes < 0.2   # al principio lo ve como legítimo
    n = agente.aprender([TEXTO] * 60, [1] * 60)   # le enseñamos que ese estilo es malicioso
    despues = agente.recibir_correo(TEXTO).probabilidad
    assert n == 60 and despues > antes + 0.3
    assert agente.modelo.ejemplos_agregados == 60


def test_actualizar_mantiene_probabilidades_validas_y_explicacion():
    agente = _agente()
    agente.aprender(["urgent verify your account"] * 5, [1] * 5)
    assert sum(agente.modelo.probabilidades.values()) == pytest.approx(1.0)
    p = agente.modelo.predecir(agente.extractor.vectorizar([TEXTO, "urgent verify your account"]))
    assert ((p >= 0) & (p <= 1)).all()
    x = agente.extractor.vectorizar(["urgent verify your password"])
    assert agente.modelo.terminos_influyentes(x, agente.extractor.vocabulario, 3)


def test_aprender_valida_las_entradas():
    agente = _agente()
    with pytest.raises(ValueError, match="una etiqueta por cada correo"):
        agente.aprender(["a b c d"], [1, 0])
    with pytest.raises(ValueError, match="0 .*1"):
        agente.aprender(["a b c d"], [2])
    with pytest.raises(ValueError, match="vacíos"):
        agente.aprender(["   "], [1])
    assert agente.aprender([], []) == 0
    assert agente.modelo.ejemplos_agregados == 0


def test_modelo_calibrado_exige_decidir_que_hacer_con_el_calibrador():
    agente = _agente(calibrar=True)
    X = agente.extractor.vectorizar([TEXTO])
    with pytest.raises(RuntimeError, match="calibrado"):
        agente.modelo.actualizar(X, np.array([1]))
    agente.aprender([TEXTO] * 5, [1] * 5, descartar_calibracion=True)
    p = agente.modelo.predecir(X)
    assert ((p >= 0) & (p <= 1)).all()


def test_guardar_conserva_lo_aprendido_y_las_metricas(tmp_path):
    agente = _agente()
    agente.metricas = {"prueba": {"exactitud": 0.9}}
    agente.aprender([TEXTO] * 40, [1] * 40)
    ruta = tmp_path / "m.joblib"
    agente.guardar(ruta)
    otro = AgenteOrquestador.desde_archivo(ruta)
    assert otro.metricas == {"prueba": {"exactitud": 0.9}}
    assert otro.modelo.ejemplos_agregados == 40
    assert otro.recibir_correo(TEXTO).probabilidad == pytest.approx(agente.recibir_correo(TEXTO).probabilidad)
