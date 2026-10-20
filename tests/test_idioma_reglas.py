"""Pruebas de la detección de idioma (D7) y de las reglas de riesgo (D8)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishing_detector.correo import CorreoElectronico  # noqa: E402
from phishing_detector.idioma import (  # noqa: E402
    ESPANOL, INDETERMINADO, INGLES, OTRO, detectar_idioma, modelo_aplicable,
)
from phishing_detector.reglas import (  # noqa: E402
    dominio_registrable, evaluar_reglas, total_puntos,
)


def _reglas(texto, remitente=""):
    correo = CorreoElectronico.desde_texto_plano(texto)
    if remitente:
        correo.remitente = remitente
    return {h.regla: h for h in evaluar_reglas(correo)}


# --------------------------------------------------------------------------- idioma
def test_idioma_ingles():
    assert detectar_idioma("Your order has shipped and will arrive on Thursday. "
                           "You can track your package in your account.") == INGLES


def test_idioma_espanol():
    assert detectar_idioma("Su pedido fue enviado y llegará el jueves. "
                           "Puede seguir el paquete desde su cuenta.") == ESPANOL


def test_idioma_otro_y_textos_cortos():
    assert detectar_idioma("Bonjour, votre compte a été suspendu. Veuillez vérifier "
                           "vos informations immédiatement.") == OTRO
    assert detectar_idioma("Click here") == INDETERMINADO


def test_idioma_ignora_enlaces_y_correos():
    assert detectar_idioma("http://example.com/a/b/c ana@empresa.com hola") == INDETERMINADO


def test_modelo_aplicable_solo_en_ingles_o_indeterminado():
    assert modelo_aplicable(INGLES) and modelo_aplicable(INDETERMINADO)
    assert not modelo_aplicable(ESPANOL) and not modelo_aplicable(OTRO)


# --------------------------------------------------------------------------- utilidades
@pytest.mark.parametrize("host,esperado", [
    ("www.empresa.com", "empresa.com"),
    ("a.b.empresa.com.co", "empresa.com.co"),
    ("news.bbc.co.uk", "bbc.co.uk"),
    ("empresa.com", "empresa.com"),
])
def test_dominio_registrable(host, esperado):
    assert dominio_registrable(host) == esperado


# --------------------------------------------------------------------------- enlaces
def test_enlace_con_ip_es_fuerte():
    h = _reglas("Entra aquí http://192.168.1.5/login ahora")
    assert h["ip_en_enlace"].puntos == 3


def test_enlace_con_arroba_es_fuerte():
    h = _reglas("Visit http://paypal.com@evil.biz/login to continue")
    assert h["arroba_en_enlace"].puntos == 3


def test_acortador_es_debil():
    h = _reglas("Look at this https://bit.ly/3abcde")
    assert h["acortador"].puntos == 1


def test_dominio_distinto_al_remitente():
    h = _reglas("Mira https://otro-sitio.net/x", remitente="Ana <ana@empresa.com>")
    assert "dominio_distinto" in h
    # el mismo dominio o un subdominio no cuentan
    assert "dominio_distinto" not in _reglas("Mira https://www.empresa.com/x", remitente="ana@empresa.com")
    assert "dominio_distinto" not in _reglas("Mira https://news.empresa.com.co/x", remitente="ana@empresa.com.co")
    # sin remitente no se puede comparar
    assert "dominio_distinto" not in _reglas("Mira https://otro-sitio.net/x")


def test_marca_en_enlace_y_dominio_sospechoso():
    h = _reglas("Please log in at http://paypal-secure-login.example.com today")
    assert h["marca_en_enlace"].puntos == 2
    assert "dominio_sospechoso" in h


@pytest.mark.parametrize("url", [
    "https://www.paypal.com/signin",
    "https://login.microsoftonline.com/common",
    "https://accounts.google.com/",
    "https://pineapple.org/recetas",
])
def test_enlaces_legitimos_no_activan_marca(url):
    assert "marca_en_enlace" not in _reglas(f"Mira {url} por favor")


# --------------------------------------------------------------------------- remitente
def test_marca_en_remitente():
    assert "marca_en_remitente" in _reglas("Hola", remitente="PayPal Support <help@gmail.com>")
    assert "marca_en_remitente" not in _reglas("Hola", remitente="PayPal <service@paypal.com>")
    assert "marca_en_remitente" not in _reglas("Hola", remitente="Amazon <ship@amazon.com>")


# --------------------------------------------------------------------------- texto (inglés y español)
def test_urgencia_amenaza_y_premio():
    h = _reglas("URGENT: your account will be suspended. You have won a gift card!")
    assert {"urgencia", "amenaza_cuenta", "premio"} <= set(h)
    h = _reglas("Último aviso: su cuenta ha sido bloqueada. Felicidades, ha ganado un premio.")
    assert {"urgencia", "amenaza_cuenta", "premio"} <= set(h)


def test_datos_sensibles():
    assert "datos_sensibles" in _reglas("Could you reply with your phone number and the code you just received by text?")
    assert "datos_sensibles" in _reglas("Please verify your account to keep access.")
    assert "datos_sensibles" in _reglas("Por favor verifique sus datos para continuar.")
    assert "datos_sensibles" in _reglas("Envíe su contraseña y su número de tarjeta.")


@pytest.mark.parametrize("texto", [
    "Hi team, attached is the agenda for tomorrow's meeting at 10am. Please review the budget slides before then. Thanks, Laura",
    "Your order 48213 has shipped and will arrive on Thursday. You can track your package in your account.",
    "Your monthly statement is now available. Log in to your account on our website to view it.",
    "Hola equipo, adjunto la agenda de la reunión de mañana. Por favor revisen el presupuesto. Gracias, Laura",
    "Su pedido 48213 fue enviado y llegará el jueves. Puede seguir el paquete desde su cuenta.",
])
def test_correos_legitimos_no_activan_reglas(texto):
    assert evaluar_reglas(CorreoElectronico.desde_texto_plano(texto)) == []


# --------------------------------------------------------------------------- combinación
def test_phishing_en_ingles_suma_mucho():
    texto = ("Your PayPal account has been limited. Please verify your identity within 24 hours at "
             "http://paypal-secure-login.example.com or your account will be suspended.")
    hallazgos = evaluar_reglas(CorreoElectronico.desde_texto_plano(texto))
    assert total_puntos(hallazgos) >= 4
    # ordenados de mayor a menor puntaje
    assert [h.puntos for h in hallazgos] == sorted((h.puntos for h in hallazgos), reverse=True)


def test_phishing_en_espanol_suma_mucho():
    texto = ("Estimado cliente, su cuenta ha sido bloqueada. Ingrese a http://banco-seguro.example.com "
             "para verificar sus datos.")
    assert total_puntos(evaluar_reglas(CorreoElectronico.desde_texto_plano(texto))) >= 4


def test_phishing_sin_enlaces_pero_pidiendo_codigo_llega_a_medio():
    texto = ("Hello, I am updating our records. Could you reply with your phone number and the code "
             "you just received by text? Regards, Support")
    assert total_puntos(evaluar_reglas(CorreoElectronico.desde_texto_plano(texto))) >= 2
