"""Prueba el modelo entrenado con correos escritos a mano.

Uso (desde la raíz del proyecto, con el entorno virtual activo y después de
`python scripts/entrenar.py`):

    python scripts/probar_ejemplos.py

Son ejemplos ilustrativos, no una medida estadística: sirven para ver cómo se
comporta el modelo fuera del corpus con el que se entrenó (docs/DISENO.md, §8).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402

from phishing_detector.config import RUTA_MODELO_ENTRENADO  # noqa: E402
from phishing_detector.evaluacion import nivel_de_riesgo  # noqa: E402

# (esperado, descripción, texto). "esperado" solo es la intención del autor.
CASOS = [
    ("malicioso", "Phishing PayPal (inglés)",
     "Your PayPal account has been limited. Please verify your identity within 24 hours at "
     "http://paypal-secure-login.example.com or your account will be suspended."),
    ("malicioso", "Phishing banco (inglés)",
     "Dear customer, we detected unusual sign-in activity on your bank account. "
     "Confirm your password here: http://bank-verify.example.net"),
    ("malicioso", "Phishing en español",
     "Estimado cliente, su cuenta ha sido bloqueada. Ingrese a http://banco-seguro.example.com "
     "para verificar sus datos."),
    ("malicioso", "Phishing sin palabras típicas (pide un código por SMS)",
     "Hello, I am updating our records. Could you reply with your phone number and the code "
     "you just received by text? Regards, Support"),
    ("legítimo", "Legítimo: reunión (inglés)",
     "Hi team, attached is the agenda for tomorrow's meeting at 10am. Please review the budget "
     "slides before then. Thanks, Laura"),
    ("legítimo", "Legítimo: pedido enviado (inglés)",
     "Your order 48213 has shipped and will arrive on Thursday. You can track your package in "
     "your account."),
    ("legítimo", "Legítimo: extracto bancario (inglés)",
     "Your monthly statement is now available. Log in to your account on our website to view it. "
     "Thank you for banking with us."),
    ("legítimo", "Legítimo: reunión (español)",
     "Hola equipo, adjunto la agenda de la reunión de mañana a las 10. Por favor revisen el "
     "presupuesto antes. Gracias, Laura"),
    ("legítimo", "Legítimo: pedido enviado (español)",
     "Su pedido 48213 fue enviado y llegará el jueves. Puede seguir el paquete desde su cuenta."),
]


def main() -> None:
    if not RUTA_MODELO_ENTRENADO.exists():
        sys.exit(f"No hay modelo en {RUTA_MODELO_ENTRENADO}. Ejecuta antes: python scripts/entrenar.py")
    guardado = joblib.load(RUTA_MODELO_ENTRENADO)
    extractor, modelo = guardado["extractor"], guardado["modelo"]

    print(f"{'P(malicioso)':>12}  {'nivel':<6} {'esperado':<10} descripción")
    for esperado, descripcion, texto in CASOS:
        p = float(modelo.predecir(extractor.vectorizar([texto]))[0])
        print(f"{p:12.3f}  {nivel_de_riesgo(p):<6} {esperado:<10} {descripcion}")


if __name__ == "__main__":
    main()
