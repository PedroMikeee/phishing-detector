"""Actualiza el modelo guardado con correos nuevos ya etiquetados (decisión D9).

Uso (desde la raíz del proyecto, con el entorno virtual activo):

    python scripts/actualizar_modelo.py data/raw/correos_nuevos.csv

El CSV usa el mismo formato que los de entrenamiento (columna de texto y
columna de etiqueta; ver config.py). Antes de incorporar los ejemplos se mide
cómo los clasificaba el modelo actual (una estimación honesta, porque todavía
no los había visto) y se guarda una copia de seguridad del modelo.

Límite: el vocabulario TF-IDF queda fijo; las palabras nuevas se ignoran. Para
ampliarlo hay que reentrenar con scripts/entrenar.py.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishing_detector.config import RUTA_MODELO_ENTRENADO  # noqa: E402
from phishing_detector.data import ConjuntoDatos  # noqa: E402
from phishing_detector.evaluacion import evaluar_probabilidades  # noqa: E402
from phishing_detector.orquestador import AgenteOrquestador  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ejemplos", help="CSV con correos nuevos etiquetados")
    ap.add_argument("--modelo", default=str(RUTA_MODELO_ENTRENADO), help="modelo de partida")
    ap.add_argument("--salida", help="dónde guardar el modelo actualizado (por defecto, sobrescribe --modelo)")
    ap.add_argument("--descartar-calibracion", action="store_true",
                    help="permite actualizar un modelo calibrado descartando su calibración")
    args = ap.parse_args()

    ruta_modelo = Path(args.modelo)
    salida = Path(args.salida) if args.salida else ruta_modelo
    agente = AgenteOrquestador.desde_archivo(ruta_modelo)

    cd = ConjuntoDatos(Path(args.ejemplos))
    df = cd.cargar_datos()
    print(f"Ejemplos nuevos: {len(df):,} ({cd.metricas['legitimos']:,} legítimos, "
          f"{cd.metricas['maliciosos']:,} maliciosos; se descartaron "
          f"{cd.metricas['duplicados_eliminados']:,} duplicados)")

    # Cómo los clasificaba el modelo actual: aún no los ha visto.
    textos, y = df["texto"].tolist(), df["etiqueta"].to_numpy()
    p = agente.modelo.predecir(agente.extractor.vectorizar(textos))
    antes = evaluar_probabilidades(y, p)
    print(f"\nAntes de aprenderlos: exactitud {antes['exactitud']:.3f}, "
          f"precisión {antes['precision']:.3f}, sensibilidad {antes['sensibilidad']:.3f}")

    if salida.exists():
        copia = salida.with_name(salida.name + ".bak")
        shutil.copy2(salida, copia)
        print(f"Copia de seguridad: {copia}")

    try:
        n = agente.aprender(textos, [int(v) for v in y], descartar_calibracion=args.descartar_calibracion)
    except RuntimeError as e:
        sys.exit(f"\nNo se pudo actualizar: {e}")

    agente.metricas = agente.metricas or {}
    agente.metricas.setdefault("actualizaciones", []).append({
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "archivo": Path(args.ejemplos).name,
        "ejemplos": n,
        "metricas_antes": {k: antes[k] for k in ("exactitud", "precision", "sensibilidad", "f1")},
    })
    agente.guardar(salida)
    print(f"\nSe incorporaron {n:,} ejemplos. Total acumulado desde el entrenamiento: "
          f"{agente.modelo.ejemplos_agregados:,}.\nModelo guardado en {salida}")


if __name__ == "__main__":
    main()
