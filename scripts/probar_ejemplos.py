"""Prueba el mecanismo con correos escritos a mano.

Uso (desde la raíz del proyecto, con el entorno virtual activo y después de
`python scripts/entrenar.py`):

    python scripts/probar_ejemplos.py

Muestra, para cada correo, la probabilidad del modelo, el nivel que daría el
modelo solo y el nivel final del mecanismo (modelo + reglas + aviso de idioma).

Son ejemplos ilustrativos, NO una medida estadística: las reglas se escribieron
mirando estos mismos casos, así que acertarlos no demuestra nada. La medida
honesta está en scripts/evaluar_mecanismo.py (docs/DISENO.md, §8).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishing_detector.alerta import NivelRiesgo  # noqa: E402
from phishing_detector.config import RUTA_MODELO_ENTRENADO  # noqa: E402
from phishing_detector.orquestador import AgenteOrquestador  # noqa: E402

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
    try:
        agente = AgenteOrquestador.desde_archivo(RUTA_MODELO_ENTRENADO)
    except FileNotFoundError as e:
        sys.exit(str(e))

    print(f"{'P(modelo)':>9}  {'solo modelo':<11} {'mecanismo':<10} {'idioma':<14} {'esperado':<10} descripción")
    aciertos_modelo = aciertos_mecanismo = 0
    for esperado, descripcion, texto in CASOS:
        p = float(agente.modelo.predecir(agente.extractor.vectorizar([texto]))[0])
        nivel_modelo = agente.evaluar_riesgo(p)
        alerta = agente.recibir_correo(texto)
        # Acierto: un correo malicioso debe salir medio o alto; uno legítimo, bajo.
        ok = lambda n: (n is not NivelRiesgo.BAJO) if esperado == "malicioso" else (n is NivelRiesgo.BAJO)  # noqa: E731
        aciertos_modelo += ok(nivel_modelo)
        aciertos_mecanismo += ok(alerta.nivel)
        print(f"{p:9.3f}  {nivel_modelo.value:<11} {alerta.nivel.value:<10} {alerta.idioma:<14} "
              f"{esperado:<10} {descripcion}")
    print(f"\nAciertos sobre {len(CASOS)} casos: solo modelo {aciertos_modelo}, mecanismo {aciertos_mecanismo} "
          "(ilustrativo: las reglas se escribieron viendo estos casos).")


if __name__ == "__main__":
    main()
