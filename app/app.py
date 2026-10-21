"""Interfaz Streamlit del mecanismo (CRISP-DM B.6, docs/DISENO.md §7).

Uso (desde la raíz del proyecto, con el entorno virtual activo y un modelo entrenado):

    streamlit run app/app.py

La interfaz solo habla con `AgenteOrquestador`: pega un correo, pulsa «Analizar»
y muestra el nivel de riesgo, los motivos y qué hacer. El análisis se hace en tu
equipo; la aplicación no guarda el texto del correo.

Para probar con otro modelo: PHISHING_MODELO=ruta/al/modelo.joblib streamlit run app/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from phishing_detector.alerta import Alerta, NivelRiesgo  # noqa: E402
from phishing_detector.config import RUTA_MODELO_ENTRENADO  # noqa: E402
from phishing_detector.idioma import NOMBRES  # noqa: E402
from phishing_detector.orquestador import AgenteOrquestador  # noqa: E402

SIN_EJEMPLO = "(escribir o pegar un correo propio)"

# Correos de demostración. No salen de ningún dataset: se escribieron a mano.
EJEMPLOS = {
    "Phishing: cuenta limitada (inglés)": (
        "Subject: Action required: your account has been limited\n\n"
        "Your PayPal account has been limited. Please verify your identity within 24 hours at "
        "http://paypal-secure-login.example.com or your account will be suspended."
    ),
    "Phishing: remitente que imita a una marca": (
        "From: PayPal Support <help@secure-paypa1-login.com>\n"
        "Subject: Unusual activity\n\n"
        "We noticed a sign-in from a new device. Review it here: "
        "http://paypal.verify-account.example.net/review"
    ),
    "Legítimo: reunión (inglés)": (
        "From: Laura <laura@empresa.example.com>\n"
        "Subject: Agenda for tomorrow\n\n"
        "Hi team, attached is the agenda for tomorrow's meeting at 10am. Please review the "
        "budget slides before then. Thanks, Laura"
    ),
    "Correo en español (el modelo no lo entiende)": (
        "Asunto: Tu pedido fue enviado\n\n"
        "Su pedido 48213 fue enviado y llegará el jueves. Puede seguir el paquete desde su cuenta."
    ),
}

# La tarjeta de color depende del nivel; el texto siempre dice el nivel (no solo el color).
TARJETA = {
    NivelRiesgo.ALTO.value: (st.error, "🚨"),
    NivelRiesgo.MEDIO.value: (st.warning, "⚠️"),
    NivelRiesgo.BAJO.value: (st.success, "✅"),
}


def porcentaje(valor: float | None) -> str:
    return "—" if valor is None else f"{valor * 100:.1f} %".replace(".", ",")


@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_agente(ruta: str, version: float) -> AgenteOrquestador:
    """`version` (fecha del archivo) hace que un modelo reentrenado se recargue solo."""
    return AgenteOrquestador.desde_archivo(Path(ruta))


def _cargar_ejemplo() -> None:
    nombre = st.session_state.get("ejemplo")
    if nombre in EJEMPLOS:
        st.session_state["texto"] = EJEMPLOS[nombre]
        st.session_state.pop("alerta", None)


def barra_lateral(agente: AgenteOrquestador) -> None:
    st.sidebar.header("Sobre este prototipo")
    st.sidebar.warning(
        "Prototipo académico. No sustituye a un filtro de correo profesional ni a tu criterio: "
        "ante la duda, verifica por otro canal."
    )
    st.sidebar.markdown(
        "- El modelo solo se entrenó con correos **en inglés**.\n"
        "- La clase «maliciosa» de sus datos mezcla spam, estafas y phishing.\n"
        "- Las reglas de enlaces, remitente y frases son heurísticas sin entrenar."
    )
    metricas = agente.metricas or {}
    prueba, externa = metricas.get("prueba"), metricas.get("externa")
    if not prueba:
        return
    st.sidebar.subheader("Métricas del modelo")
    filas = ["| | Su corpus | Externa |", "|---|---|---|"]
    for clave, nombre in (("exactitud", "Exactitud"), ("precision", "Precisión"),
                          ("sensibilidad", "Sensibilidad"), ("auc_roc", "AUC-ROC")):
        ext = porcentaje(externa[clave]) if externa and externa.get(clave) is not None else "—"
        filas.append(f"| {nombre} | {porcentaje(prueba.get(clave))} | {ext} |")
    st.sidebar.markdown("\n".join(filas))
    nota = f"Corpus propio: {prueba['n']:,} correos de prueba".replace(",", ".")
    if externa:
        nota += f"; validación externa: {externa['n']:,} correos únicos".replace(",", ".")
    st.sidebar.caption(
        nota + ". La caída en la validación externa muestra que el modelo aprendió rasgos de sus "
        "datos de entrenamiento: las cifras del corpus propio son optimistas."
    )


def mostrar_alerta(alerta: Alerta) -> None:
    dibujar, icono = TARJETA[alerta.nivel.value]
    dibujar(f"**{alerta.mensaje}**", icon=icono)

    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad del modelo", "No usada" if alerta.probabilidad is None
              else porcentaje(alerta.probabilidad))
    c2.metric("Puntos de las reglas", alerta.puntos_reglas)
    c3.metric("Idioma detectado", NOMBRES.get(alerta.idioma, alerta.idioma))

    for aviso in alerta.advertencias:
        st.warning(aviso)

    if alerta.motivos:
        st.subheader("Por qué")
        st.markdown("\n".join(f"- {m}" for m in alerta.motivos))
    # Las palabras solo explican cuando el modelo ve riesgo; en un nivel bajo confundirían.
    if alerta.terminos_sospechosos and alerta.nivel_modelo not in (None, NivelRiesgo.BAJO):
        palabras = ", ".join(f"`{t}`" for t, _ in alerta.terminos_sospechosos)
        st.markdown(f"**Palabras que más pesaron en el modelo:** {palabras}")
    if alerta.enlaces_detectados:
        st.markdown("**Enlaces detectados** (no los abras):")
        st.code("\n".join(alerta.enlaces_detectados), language=None)   # texto plano: no se puede hacer clic

    st.subheader("Qué hacer")
    st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(alerta.acciones_sugeridas, 1)))

    with st.expander("Cómo se calculó este nivel"):
        nivel_modelo = "no se usó" if alerta.nivel_modelo is None or alerta.probabilidad is None \
            else alerta.nivel_modelo.value
        st.markdown(
            f"- Nivel según el modelo: **{nivel_modelo}**.\n"
            f"- Puntos de las reglas: **{alerta.puntos_reglas}** (2 o más = medio, 4 o más = alto).\n"
            f"- Nivel final: **{alerta.nivel.value}**, el más grave de los dos. "
            "Las reglas solo pueden subir el nivel, nunca bajarlo."
        )


def main() -> None:
    st.set_page_config(page_title="Detector de phishing", page_icon="🛡️", layout="centered")
    st.title("Detector de phishing en correos")
    st.caption("Pega el texto de un correo para estimar su riesgo y saber cómo actuar.")

    ruta = Path(os.environ.get("PHISHING_MODELO", str(RUTA_MODELO_ENTRENADO)))
    if not ruta.exists():
        st.error(f"No hay un modelo entrenado en `{ruta}`.")
        st.code("python scripts/entrenar.py", language="bash")
        st.stop()
    agente = cargar_agente(str(ruta), ruta.stat().st_mtime)
    barra_lateral(agente)

    st.selectbox("Cargar un ejemplo", [SIN_EJEMPLO, *EJEMPLOS], key="ejemplo", on_change=_cargar_ejemplo)
    texto = st.text_area(
        "Correo (puedes empezar con líneas «From:» y «Subject:»)", height=260, key="texto",
        placeholder="Pega aquí el correo…",
    )

    if st.button("Analizar", type="primary"):
        if not texto.strip():
            st.session_state.pop("alerta", None)
            st.warning("Escribe o pega un correo antes de analizar.")
        else:
            st.session_state["alerta"] = agente.recibir_correo(texto)
            agente.modulo_alerta.registro.clear()   # el modelo se comparte entre sesiones: no acumular

    if "alerta" in st.session_state:
        st.divider()
        mostrar_alerta(st.session_state["alerta"])


main()
