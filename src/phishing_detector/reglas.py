"""Reglas complementarias de riesgo (decisión D8 de docs/DISENO.md).

El modelo de palabras no ve el remitente ni los enlaces (el dataset de
entrenamiento llegó sin URLs). Estas reglas cubren esa parte del Objetivo 1 del
anteproyecto: enlaces, remitente, lenguaje de urgencia y petición de datos.

Cada regla que se cumple produce un `Hallazgo` con una explicación en lenguaje
claro y unos puntos (1 débil, 2 media, 3 fuerte). El orquestador suma los
puntos y fija un nivel mínimo de riesgo; las reglas solo pueden SUBIR el nivel
que da el modelo, nunca bajarlo. Son heurísticas, no un modelo entrenado: sus
umbrales (ConfigOrquestador.puntos_*) son un punto de partida.

Las reglas de texto reconocen frases en inglés y en español, a diferencia del
modelo, que solo conoce inglés.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from .correo import PATRON_URL, CorreoElectronico

DEBIL, MEDIA, FUERTE = 1, 2, 3


@dataclass(frozen=True)
class Hallazgo:
    regla: str          # identificador estable (se usa en las pruebas)
    descripcion: str    # motivo en lenguaje claro, para mostrar al usuario
    puntos: int


# --------------------------------------------------------------------------- utilidades de URL
_PATRON_ESQUEMA = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_PATRON_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_PATRON_DIRECCION = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")
_SEGUNDO_NIVEL = frozenset({"co", "com", "org", "net", "gov", "edu", "ac", "gob", "mil"})

ACORTADORES = frozenset({
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly", "cutt.ly",
    "rb.gy", "shorturl.at", "tiny.cc", "rebrand.ly", "lnkd.in",
})

# Palabras que aparecen en dominios falsos que imitan un acceso o una verificación.
_PALABRAS_DOMINIO = ("secure", "verify", "login", "signin", "account", "update", "confirm",
                     "support", "seguro", "verifica", "cuenta", "acceso")

# Marcas frecuentemente suplantadas -> etiquetas de dominio oficiales
# (la parte del nombre sin la extensión, p. ej. "microsoftonline" de microsoftonline.com).
MARCAS = {
    "paypal": {"paypal"},
    "microsoft": {"microsoft", "microsoftonline", "office", "office365", "live", "outlook", "sharepoint"},
    "apple": {"apple", "icloud"},
    "amazon": {"amazon", "amazonaws", "amazonses"},
    "google": {"google", "gmail", "googleapis", "gstatic", "youtube", "googleusercontent"},
    "netflix": {"netflix"},
    "facebook": {"facebook", "facebookmail", "fb"},
    "instagram": {"instagram"},
    "whatsapp": {"whatsapp"},
    "dhl": {"dhl"},
    "fedex": {"fedex"},
    "binance": {"binance"},
    "coinbase": {"coinbase"},
    "bancolombia": {"bancolombia", "grupobancolombia"},
    "davivienda": {"davivienda"},
    "nequi": {"nequi"},
    "bbva": {"bbva"},
    "santander": {"santander"},
    "itau": {"itau"},
}


def _anfitrion(url: str) -> tuple[str, bool]:
    """(dominio o IP del enlace, si el enlace usa '@' para disfrazar el destino)."""
    if not _PATRON_ESQUEMA.match(url):
        url = "http://" + url
    try:
        netloc = urlsplit(url).netloc.lower()
    except ValueError:
        return "", False
    tiene_arroba = "@" in netloc
    host = netloc.rsplit("@", 1)[-1]
    host = host.split("]")[0] + "]" if host.startswith("[") else host.split(":")[0]
    return host.strip("."), tiene_arroba


def dominio_registrable(host: str) -> str:
    """Aproximación al dominio registrable: las dos últimas etiquetas, o tres
    si termina en un dominio de segundo nivel como `com.co` o `co.uk`."""
    etiquetas = [e for e in host.split(".") if e]
    if len(etiquetas) <= 2:
        return ".".join(etiquetas)
    if len(etiquetas[-1]) == 2 and etiquetas[-2] in _SEGUNDO_NIVEL:
        return ".".join(etiquetas[-3:])
    return ".".join(etiquetas[-2:])


def _dominio_remitente(remitente: str) -> str:
    m = _PATRON_DIRECCION.search(remitente or "")
    return m.group(1).lower() if m else ""


def _nombre_visible(remitente: str) -> str:
    """Texto antes de '<' en `Nombre <a@b.com>`; vacío si no hay."""
    if "<" not in (remitente or ""):
        return ""
    return remitente.split("<", 1)[0].strip().strip('"').lower()


def _marca_suplantada(texto: str, dominio: str) -> str | None:
    """Marca mencionada en `texto` cuyo dominio oficial no es `dominio`."""
    etiqueta = dominio_registrable(dominio).split(".")[0]
    for marca, oficiales in MARCAS.items():
        # palabra completa: "pineapple.org" no cuenta como "apple"
        if re.search(rf"(?<![a-z0-9]){marca}(?![a-z0-9])", texto) and etiqueta not in oficiales:
            return marca
    return None


# --------------------------------------------------------------------------- reglas de texto
_FLAGS = re.IGNORECASE

_URGENCIA = re.compile(
    r"\b(?:urgent(?:ly)?|immediately|right away|act now|final notice|last warning|last chance|"
    r"within \d+ (?:hours?|days?)|expires? (?:today|soon|in \d+)|"
    r"urgente(?:mente)?|inmediatamente|de inmediato|(?:en|dentro de) (?:las )?(?:próximas )?\d+ horas|"
    r"último aviso|última oportunidad|caduca|vence hoy)\b", _FLAGS)

_ESTADO_CUENTA = (r"suspend\w*|locked|blocked|limited|closed|disabled|compromised|terminated|deactivated|"
                  r"suspendid\w+|bloquead\w+|limitad\w+|cerrad\w+|desactivad\w+|comprometid\w+")
_AMENAZA_CUENTA = re.compile(
    rf"\b(?:account|cuenta)\b[^.\n]{{0,60}}\b(?:{_ESTADO_CUENTA})\b|"
    rf"\b(?:{_ESTADO_CUENTA})\b[^.\n]{{0,30}}\b(?:account|cuenta)\b", _FLAGS)

_VERBOS_PEDIR = (r"send|reply with|provide|confirm|verify|enter|share|give|type|submit|update|"
                 r"envíe|envie|responda|proporcione|confirme|verifique|ingrese|comparta|indique|"
                 r"introduzca|actualice")
_DATOS = (r"password|passcode|pin|(?:verification|security|one[- ]time|authentication|confirmation|sms|access) code|"
          r"code you (?:just )?(?:received|got)|ssn|social security(?: number)?|card number|credit card|debit card|"
          r"cvv|bank details|bank account|account number|personal (?:information|details|data)|"
          r"login (?:details|credentials)|credentials|date of birth|contraseña|"
          r"código de (?:verificación|seguridad|confirmación|acceso)|código que (?:te|le) (?:llegó|enviamos|enviaron)|"
          r"número de tarjeta|tarjeta de (?:crédito|débito)|datos (?:personales|bancarios|de acceso)|"
          r"(?:su|tu) clave|clave (?:dinámica|secreta|de acceso)|(?:sus|tus) datos|información personal")
_PEDIR_DATOS = re.compile(rf"\b(?:{_VERBOS_PEDIR})\b[^.\n]{{0,60}}\b(?:{_DATOS})\b", _FLAGS)
_VERIFICAR_CUENTA = re.compile(
    r"\b(?:verify|confirm|validate|update|reactivate|restore|secure|unlock)\s+(?:your|the)\s+"
    r"(?:account|identity|information|details|payment|billing|credentials)\b|"
    r"\b(?:verifi\w*|confirm\w*|valid\w*|actualiz\w*|reactiv\w*|restablec\w*|desbloque\w*)\s+"
    r"(?:su|sus|tu|tus)\s+(?:cuenta|identidad|información|datos|pago|clave)\b", _FLAGS)

_PREMIO = re.compile(
    r"\b(?:you(?:'ve| have)? won|congratulations|claim your (?:prize|reward|gift)|gift card|free gift|"
    r"lottery|ganaste|has ganado|ha ganado|felicidades|reclama tu premio|tarjeta de regalo|lotería)\b", _FLAGS)


# --------------------------------------------------------------------------- evaluación
def _reglas_de_enlaces(correo: CorreoElectronico, enlaces: list[str]) -> list[Hallazgo]:
    hallazgos: list[Hallazgo] = []
    dom_remitente = _dominio_remitente(correo.remitente)
    hosts = []
    for url in enlaces:
        host, arroba = _anfitrion(url)
        if host:
            hosts.append(host)
        if arroba:
            hallazgos.append(Hallazgo("arroba_en_enlace",
                                      "Un enlace usa «@» para disfrazar a dónde lleva realmente.", FUERTE))
    hosts = list(dict.fromkeys(hosts))  # sin repetidos, conservando el orden

    if any(_PATRON_IPV4.match(h) or h.startswith("[") for h in hosts):
        hallazgos.append(Hallazgo("ip_en_enlace",
                                  "Un enlace apunta a una dirección numérica (IP) y no a un nombre de sitio.", FUERTE))
    if any(h.startswith("xn--") or ".xn--" in h for h in hosts):
        hallazgos.append(Hallazgo("punycode",
                                  "Un enlace usa caracteres internacionales que pueden imitar a otro sitio.", MEDIA))
    cortos = [h for h in hosts if h in ACORTADORES]
    if cortos:
        hallazgos.append(Hallazgo("acortador",
                                  f"Un enlace usa un acortador ({cortos[0]}): no se ve el destino real.", DEBIL))

    normales = [h for h in hosts if h not in ACORTADORES and not _PATRON_IPV4.match(h) and not h.startswith("[")]
    for h in normales:
        marca = _marca_suplantada(h, h)
        if marca:
            hallazgos.append(Hallazgo(
                "marca_en_enlace",
                f"Un enlace menciona «{marca}», pero el sitio real es {dominio_registrable(h)}.", MEDIA))
            break
    if any("-" in h and any(p in h for p in _PALABRAS_DOMINIO) for h in normales):
        hallazgos.append(Hallazgo("dominio_sospechoso",
                                  "Un enlace usa un dominio con guiones y palabras como «secure» o «login».", DEBIL))
    if dom_remitente:
        reg_rem = dominio_registrable(dom_remitente)
        distintos = list(dict.fromkeys(dominio_registrable(h) for h in normales
                                       if dominio_registrable(h) != reg_rem))
        if distintos:
            hallazgos.append(Hallazgo(
                "dominio_distinto",
                f"Los enlaces apuntan a {', '.join(distintos[:2])} y no al dominio del remitente ({reg_rem}).", DEBIL))
    return hallazgos


def _reglas_de_remitente(correo: CorreoElectronico) -> list[Hallazgo]:
    dom = _dominio_remitente(correo.remitente)
    nombre = _nombre_visible(correo.remitente)
    if dom and nombre:
        marca = _marca_suplantada(nombre, dom)
        if marca:
            return [Hallazgo("marca_en_remitente",
                             f"El remitente dice ser «{marca}», pero el correo viene de {dominio_registrable(dom)}.",
                             MEDIA)]
    return []


def _reglas_de_texto(contenido: str) -> list[Hallazgo]:
    hallazgos: list[Hallazgo] = []
    if _URGENCIA.search(contenido):
        hallazgos.append(Hallazgo("urgencia", "Mete prisa o plazos («urgente», «en 24 horas»).", DEBIL))
    if _AMENAZA_CUENTA.search(contenido):
        hallazgos.append(Hallazgo("amenaza_cuenta", "Amenaza con suspender o bloquear tu cuenta.", DEBIL))
    if _PEDIR_DATOS.search(contenido) or _VERIFICAR_CUENTA.search(contenido):
        hallazgos.append(Hallazgo("datos_sensibles",
                                  "Te pide verificar tu cuenta o entregar datos sensibles (clave, código, tarjeta).", MEDIA))
    if _PREMIO.search(contenido):
        hallazgos.append(Hallazgo("premio", "Promete premios o regalos.", DEBIL))
    return hallazgos


def evaluar_reglas(correo: CorreoElectronico, enlaces: list[str] | None = None) -> list[Hallazgo]:
    """Aplica todas las reglas. `enlaces` por defecto son los del correo, o los
    que se encuentren en el texto si el correo no los trae separados."""
    contenido = correo.obtener_contenido()
    if enlaces is None:
        enlaces = correo.enlaces or [u.rstrip(".,;:") for u in PATRON_URL.findall(contenido)]
    hallazgos = (_reglas_de_enlaces(correo, enlaces) + _reglas_de_remitente(correo)
                 + _reglas_de_texto(contenido))
    return sorted(hallazgos, key=lambda h: -h.puntos)


def total_puntos(hallazgos: list[Hallazgo]) -> int:
    return sum(h.puntos for h in hallazgos)
