"""Pruebas de la interfaz Streamlit (app/app.py) con el simulador AppTest.

Usan un modelo mínimo entrenado con datos sintéticos, guardado en un archivo
temporal, así que no necesitan los CSV ni el modelo real.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phishing_detector.caracteristicas import ExtractorCaracteristicas  # noqa: E402
from phishing_detector.config import ConfigModelo  # noqa: E402
from phishing_detector.modelo import ModeloNaiveBayes  # noqa: E402

APP = RAIZ / "app" / "app.py"
LEGIT = ["meeting", "project", "lunch", "report", "schedule", "thanks", "review", "team", "draft", "budget"]
MALO = ["verify", "account", "urgent", "click", "winner", "bank", "password", "prize", "login", "offer"]


@pytest.fixture(scope="module")
def modelo_temporal(tmp_path_factory):
    cfg = ConfigModelo(min_df=1)
    rng = np.random.default_rng(0)
    textos, y = [], []
    for _ in range(100):
        textos.append(" ".join(rng.choice(LEGIT, 12))); y.append(0)
        textos.append(" ".join(rng.choice(MALO, 12))); y.append(1)
    ex = ExtractorCaracteristicas(cfg)
    ex.ajustar(textos)
    modelo = ModeloNaiveBayes(cfg)
    modelo.entrenar(ex.vectorizar(textos), np.array(y))
    metricas = {"prueba": {"n": 40, "exactitud": 0.95, "precision": 0.96, "sensibilidad": 0.94, "auc_roc": 0.99},
                "externa": {"n": 10, "exactitud": 0.8, "precision": 0.5, "sensibilidad": 0.9, "auc_roc": 0.9}}
    ruta = tmp_path_factory.mktemp("modelo") / "modelo.joblib"
    joblib.dump({"extractor": ex, "modelo": modelo, "config": cfg, "metricas": metricas}, ruta)
    return ruta


@pytest.fixture
def app(modelo_temporal, monkeypatch):
    monkeypatch.setenv("PHISHING_MODELO", str(modelo_temporal))
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _analizar(at, texto):
    at.text_area(key="texto").set_value(texto)
    at.button[0].click()
    at.run()
    assert not at.exception
    return at


def _tarjetas(at):
    """Textos de las tarjetas de nivel (error/warning/success) del cuerpo principal."""
    return [e.value for lista in (at.main.error, at.main.warning, at.main.success) for e in lista]


def test_arranca_sin_resultado(app):
    assert "alerta" not in app.session_state
    assert any("Prototipo académico" in e.value for e in app.sidebar.warning)


def test_barra_lateral_muestra_metricas_con_coma_decimal(app):
    tablas = [m.value for m in app.sidebar.markdown if "Exactitud" in m.value]
    assert tablas and "95,0 %" in tablas[0] and "80,0 %" in tablas[0]


def test_correo_vacio_pide_texto(app):
    _analizar(app, "   ")
    assert any("antes de analizar" in t for t in _tarjetas(app))
    assert "alerta" not in app.session_state


def test_remitente_falso_da_riesgo_alto_por_las_reglas(app):
    """El nivel alto sale de las reglas, así que no depende del modelo sintético."""
    _analizar(app, "From: PayPal Support <help@secure-paypa1-login.com>\nSubject: Unusual activity\n\n"
                   "Review it here: http://paypal.verify-account.example.net/review")
    assert any("Riesgo ALTO" in e.value for e in app.main.error)
    assert any("Por qué" in s.value for s in app.main.subheader)
    assert any("no los abras" in m.value for m in app.main.markdown)


def test_correo_en_espanol_avisa_que_no_usa_el_modelo(app):
    _analizar(app, "Hola equipo, adjunto la agenda de la reunión de mañana a las 10. "
                   "Por favor revisen el presupuesto antes de la reunión. Gracias.")
    assert any("solo se entrenó con correos en inglés" in w.value for w in app.main.warning)
    assert any(m.label == "Probabilidad del modelo" and m.value == "No usada" for m in app.main.metric)


def test_correo_legitimo_en_ingles_da_riesgo_bajo(app):
    _analizar(app, "meeting project lunch report schedule thanks review team draft budget meeting")
    assert any("Riesgo BAJO" in e.value for e in app.main.success)


def test_cargar_ejemplo_llena_la_caja_de_texto(app):
    app.selectbox(key="ejemplo").select("Phishing: cuenta limitada (inglés)").run()
    assert not app.exception
    assert "PayPal account has been limited" in app.text_area(key="texto").value


def test_todos_los_ejemplos_se_analizan_sin_error(app):
    for nombre in app.selectbox(key="ejemplo").options[1:]:
        app.selectbox(key="ejemplo").select(nombre).run()
        app.button[0].click().run()
        assert not app.exception, nombre
        assert "alerta" in app.session_state


def test_sin_modelo_muestra_como_entrenarlo(monkeypatch, tmp_path):
    monkeypatch.setenv("PHISHING_MODELO", str(tmp_path / "no_existe.joblib"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("No hay un modelo entrenado" in e.value for e in at.error)
    assert any("scripts/entrenar.py" in c.value for c in at.code)
