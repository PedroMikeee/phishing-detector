"""Pruebas de ModuloAlerta y AgenteOrquestador con un modelo sintético."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishing_detector.alerta import (  # noqa: E402
    ACCIONES, MENSAJE_BAJO_SIN_MODELO, MENSAJES, ModuloAlerta, NivelRiesgo, nivel_maximo,
)
from phishing_detector.caracteristicas import ExtractorCaracteristicas  # noqa: E402
from phishing_detector.config import ConfigModelo, ConfigOrquestador  # noqa: E402
from phishing_detector.correo import CorreoElectronico  # noqa: E402
from phishing_detector.modelo import ModeloNaiveBayes  # noqa: E402
from phishing_detector.orquestador import AgenteOrquestador  # noqa: E402

LEGIT = ["meeting", "project", "lunch", "report", "schedule", "thanks", "review", "team", "draft", "budget"]
MALO = ["verify", "account", "urgent", "click", "winner", "bank", "password", "prize", "login", "offer"]

# Frases en inglés con suficientes palabras vacías para que se detecte el idioma.
TEXTO_MALO = "please click the link to verify your account and claim the prize and the offer"
TEXTO_LEGIT = "the team will review the budget and the report for the project at the meeting"
TEXTO_ES_PHISHING = ("Estimado cliente, su cuenta ha sido bloqueada. Ingrese a "
                     "http://banco-seguro.example.com para verificar sus datos.")
TEXTO_ES_LEGIT = "Hola equipo, adjunto la agenda de la reunión de mañana. Gracias, Laura"


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


@pytest.fixture
def agente(entrenado):
    ex, modelo = entrenado
    return AgenteOrquestador(ex, modelo)


def test_sin_verificar_idioma_se_usa_el_modelo_aunque_no_parezca_ingles(entrenado):
    """Para corpus sin palabras vacías (ver evaluar_mecanismo.py) el filtro de idioma se apaga."""
    ex, modelo = entrenado
    texto = "verify account password login bank prize winner click offer urgent verify account bank"
    normal = AgenteOrquestador(ex, modelo).recibir_correo(texto)
    sin_filtro = AgenteOrquestador(ex, modelo, cfg=ConfigOrquestador(verificar_idioma=False)).recibir_correo(texto)
    assert normal.probabilidad is None and normal.advertencias          # sin palabras vacías: "otro idioma"
    assert sin_filtro.probabilidad is not None and not sin_filtro.advertencias
    assert sin_filtro.nivel is NivelRiesgo.ALTO


# --------------------------------------------------------------------------- alerta
def test_niveles_se_ordenan_y_nivel_maximo():
    assert NivelRiesgo.BAJO.orden < NivelRiesgo.MEDIO.orden < NivelRiesgo.ALTO.orden
    assert nivel_maximo(NivelRiesgo.BAJO, NivelRiesgo.ALTO, NivelRiesgo.MEDIO) is NivelRiesgo.ALTO
    assert nivel_maximo(NivelRiesgo.MEDIO) is NivelRiesgo.MEDIO


@pytest.mark.parametrize("nivel", list(NivelRiesgo))
def test_construir_alerta_usa_textos_del_diseno(nivel):
    alerta = ModuloAlerta().construir_alerta(nivel, 0.5, [], [])
    assert alerta.mensaje == MENSAJES[nivel]
    assert alerta.acciones_sugeridas == ACCIONES[nivel]


def test_bajo_sin_modelo_no_se_presenta_como_seguro():
    alerta = ModuloAlerta().construir_alerta(NivelRiesgo.BAJO, None, [], [])
    assert alerta.mensaje == MENSAJE_BAJO_SIN_MODELO
    assert len(alerta.acciones_sugeridas) == len(ACCIONES[NivelRiesgo.BAJO]) + 1


def test_mostrar_alerta_formatea_porcentaje_y_secciones():
    modulo = ModuloAlerta()
    alerta = modulo.construir_alerta(
        NivelRiesgo.ALTO, 0.973, [("verify", 2.0)], ["http://x.example"],
        motivos=["Mete prisa."], advertencias=["Aviso de prueba."], nivel_modelo=NivelRiesgo.ALTO)
    texto = modulo.mostrar_alerta(alerta)
    assert texto.startswith("ALTO — probabilidad del modelo: 97,3 %")
    for esperado in ("Por qué:", "Mete prisa.", "Palabras que más pesaron en el modelo: verify",
                     "http://x.example", "Qué hacer:", "1. No hagas clic", "Avisos:", "Aviso de prueba."):
        assert esperado in texto


def test_mostrar_alerta_omite_palabras_si_el_modelo_ve_riesgo_bajo():
    modulo = ModuloAlerta()
    alerta = modulo.construir_alerta(NivelRiesgo.BAJO, 0.1, [("verify", 0.2)], [],
                                     nivel_modelo=NivelRiesgo.BAJO)
    assert "Palabras que más pesaron" not in modulo.mostrar_alerta(alerta)


# --------------------------------------------------------------------------- orquestador: modelo
def test_umbrales_de_riesgo(agente):
    assert agente.evaluar_riesgo(0.34) is NivelRiesgo.BAJO
    assert agente.evaluar_riesgo(0.35) is NivelRiesgo.MEDIO
    assert agente.evaluar_riesgo(0.69) is NivelRiesgo.MEDIO
    assert agente.evaluar_riesgo(0.70) is NivelRiesgo.ALTO


def test_correo_malicioso_en_ingles_es_alto(agente):
    alerta = agente.recibir_correo(TEXTO_MALO)
    assert alerta.nivel is NivelRiesgo.ALTO and alerta.nivel_modelo is NivelRiesgo.ALTO
    assert 0.7 <= alerta.probabilidad <= 1.0
    assert alerta.terminos_sospechosos
    assert all(set(t.split()) <= set(MALO) | set(TEXTO_MALO.split()) for t, _ in alerta.terminos_sospechosos)
    assert alerta.acciones_sugeridas == ACCIONES[NivelRiesgo.ALTO]
    assert alerta.advertencias == []


def test_correo_legitimo_en_ingles_es_bajo(agente):
    alerta = agente.recibir_correo(TEXTO_LEGIT)
    assert alerta.nivel is NivelRiesgo.BAJO
    assert alerta.probabilidad < 0.35
    assert alerta.motivos == []


def test_acepta_correo_electronico_y_texto_con_cabeceras(agente):
    a = agente.recibir_correo(CorreoElectronico(asunto="Urgent", cuerpo=TEXTO_MALO))
    b = agente.recibir_correo(f"Subject: Urgent\n{TEXTO_MALO}")
    assert a.nivel is b.nivel is NivelRiesgo.ALTO
    assert a.probabilidad == pytest.approx(b.probabilidad)


def test_correo_vacio_falla_con_mensaje_claro(agente):
    with pytest.raises(ValueError, match="vacío"):
        agente.recibir_correo("   \n ")


def test_registra_el_evento_y_el_ultimo_nivel(agente):
    assert agente.nivel_alerta is None
    alerta = agente.recibir_correo(TEXTO_MALO)
    assert agente.nivel_alerta is alerta.nivel
    assert agente.modulo_alerta.registro == [alerta]


def test_generar_recomendacion_devuelve_una_copia(agente):
    acciones = agente.generar_recomendacion(NivelRiesgo.ALTO)
    acciones.append("algo")
    assert agente.generar_recomendacion(NivelRiesgo.ALTO) == ACCIONES[NivelRiesgo.ALTO]


# --------------------------------------------------------------------------- orquestador: reglas
def test_las_reglas_suben_el_nivel_cuando_el_modelo_no_ve_riesgo(agente):
    texto = TEXTO_LEGIT + " and reply with your password and the code you just received"
    alerta = agente.recibir_correo(texto)
    assert alerta.nivel_modelo is NivelRiesgo.BAJO
    assert alerta.nivel is NivelRiesgo.MEDIO
    assert alerta.puntos_reglas >= 2 and alerta.motivos


def test_las_reglas_nunca_bajan_el_nivel_del_modelo(agente):
    # palabras maliciosas para el modelo, pero sin enlaces ni frases que activen reglas
    alerta = agente.recibir_correo("the winner of the prize can click the offer and login to the bank")
    assert alerta.puntos_reglas == 0
    assert alerta.nivel is alerta.nivel_modelo is NivelRiesgo.ALTO


def test_enlaces_se_extraen_del_texto(agente):
    alerta = agente.recibir_correo(TEXTO_MALO + " http://192.168.0.7/login")
    assert alerta.enlaces_detectados == ["http://192.168.0.7/login"]
    assert any("dirección numérica" in m for m in alerta.motivos)


# --------------------------------------------------------------------------- orquestador: idioma
def test_phishing_en_espanol_no_usa_el_modelo_pero_las_reglas_lo_detectan(agente):
    alerta = agente.recibir_correo(TEXTO_ES_PHISHING)
    assert alerta.idioma == "es"
    assert alerta.probabilidad is None and alerta.terminos_sospechosos == []
    assert alerta.nivel is NivelRiesgo.ALTO
    assert alerta.advertencias and "español" in alerta.advertencias[0]


def test_correo_legitimo_en_espanol_no_alarma_ni_se_presenta_como_seguro(agente):
    alerta = agente.recibir_correo(TEXTO_ES_LEGIT)
    assert alerta.nivel is NivelRiesgo.BAJO and alerta.probabilidad is None
    assert alerta.mensaje == MENSAJE_BAJO_SIN_MODELO
    assert "Avisos:" in agente.modulo_alerta.mostrar_alerta(alerta)


# --------------------------------------------------------------------------- carga desde archivo
def test_desde_archivo_da_las_mismas_predicciones(entrenado, tmp_path):
    ex, modelo = entrenado
    ruta = tmp_path / "modelo.joblib"
    joblib.dump({"extractor": ex, "modelo": modelo}, ruta)
    cargado = AgenteOrquestador.desde_archivo(ruta)
    original = AgenteOrquestador(ex, modelo)
    assert cargado.recibir_correo(TEXTO_MALO).probabilidad == pytest.approx(
        original.recibir_correo(TEXTO_MALO).probabilidad)


def test_desde_archivo_sin_modelo_explica_que_hacer(tmp_path):
    with pytest.raises(FileNotFoundError, match="entrenar.py"):
        AgenteOrquestador.desde_archivo(tmp_path / "no_existe.joblib")
