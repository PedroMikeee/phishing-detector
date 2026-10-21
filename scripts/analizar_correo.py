"""Analiza un correo y muestra la alerta, sin interfaz gráfica.

Uso (desde la raíz del proyecto, con el entorno virtual activo y un modelo entrenado):

    python scripts/analizar_correo.py --texto "Your account has been suspended. Verify at http://x-secure.example.com"
    python scripts/analizar_correo.py correo.txt
    cat correo.txt | python scripts/analizar_correo.py

El texto puede empezar con cabeceras `From:` y `Subject:` (o `De:` y `Asunto:`):
si el remitente viene en el texto, las reglas también comparan el dominio de los enlaces.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishing_detector.config import RUTA_MODELO_ENTRENADO  # noqa: E402
from phishing_detector.orquestador import AgenteOrquestador  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archivo", nargs="?", help="archivo de texto con el correo (si no, usa --texto o la entrada estándar)")
    ap.add_argument("--texto", help="texto del correo")
    ap.add_argument("--modelo", default=str(RUTA_MODELO_ENTRENADO))
    args = ap.parse_args()

    if args.texto:
        texto = args.texto
    elif args.archivo:
        texto = Path(args.archivo).read_text(encoding="utf-8", errors="replace")
    elif not sys.stdin.isatty():
        texto = sys.stdin.read()
    else:
        ap.error("indica un archivo, --texto o envía el correo por la entrada estándar")

    try:
        agente = AgenteOrquestador.desde_archivo(Path(args.modelo))
        alerta = agente.recibir_correo(texto)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    print(agente.modulo_alerta.mostrar_alerta(alerta))


if __name__ == "__main__":
    main()
