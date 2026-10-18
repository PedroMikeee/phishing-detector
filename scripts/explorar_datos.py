"""Exploración de datos (CRISP-DM B.2 - Comprensión de los datos).

Uso (desde la raíz del proyecto, con el entorno virtual activo):

    python scripts/explorar_datos.py                      # usa el primer CSV de data/raw/
    python scripts/explorar_datos.py data/raw/CEAS_08.csv

Imprime un resumen en consola, guarda dos figuras en docs/figuras/ y un
reporte JSON en data/processed/reporte_exploratorio.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402

from phishing_detector.caracteristicas import ExtractorCaracteristicas  # noqa: E402
from phishing_detector.config import RAIZ, RUTA_DATOS_CRUDOS, RUTA_DATOS_PROCESADOS  # noqa: E402
from phishing_detector.data import ConjuntoDatos  # noqa: E402

# Paleta validada (azul y naranja: separación para daltonismo y contraste OK).
COLOR_LEGITIMO = "#2a78d6"
COLOR_MALICIOSO = "#eb6834"
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
CUADRICULA = "#e1e0d9"
NOMBRES = {0: "Legítimo", 1: "Malicioso"}
COLORES = {0: COLOR_LEGITIMO, 1: COLOR_MALICIOSO}


def _estilo(ax):
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c3c2b7")
    ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=9)
    ax.yaxis.grid(True, color=CUADRICULA, linewidth=0.8)
    ax.set_axisbelow(True)


def figura_balance(df, ruta):
    conteo = df["etiqueta"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.6), facecolor=SUPERFICIE)
    _estilo(ax)
    total = conteo.sum()
    for i, (clase, n) in enumerate(conteo.items()):
        ax.bar(i, n, width=0.55, color=COLORES[clase])
        ax.text(i, n, f"{n:,}\n({n / total:.1%})", ha="center", va="bottom",
                fontsize=9, color=TINTA)
    ax.set_xticks(range(len(conteo)))
    ax.set_xticklabels([NOMBRES[c] for c in conteo.index], color=TINTA)
    ax.set_ylim(0, conteo.max() * 1.22)
    ax.set_ylabel("Número de correos", color=TINTA_SECUNDARIA, fontsize=9)
    ax.set_title("Balance de clases", color=TINTA, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, facecolor=SUPERFICIE)
    plt.close(fig)


def figura_longitud(df, ruta):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SUPERFICIE)
    _estilo(ax)
    longitudes = df["texto"].str.len().clip(lower=1)
    bordes = np.logspace(np.log10(longitudes.min()), np.log10(max(longitudes.max(), longitudes.min() * 10)), 40)
    for clase in (0, 1):
        ax.hist(longitudes[df["etiqueta"] == clase], bins=bordes, histtype="step",
                linewidth=2, color=COLORES[clase], label=NOMBRES[clase])
    ax.set_xscale("log")
    ax.set_xlabel("Longitud del correo (caracteres, escala logarítmica)",
                  color=TINTA_SECUNDARIA, fontsize=9)
    ax.set_ylabel("Número de correos", color=TINTA_SECUNDARIA, fontsize=9)
    ax.set_title("Distribución de la longitud por clase", color=TINTA, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TINTA)
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, facecolor=SUPERFICIE)
    plt.close(fig)


def terminos_frecuentes(df, clase, n=15, muestra=8000):
    """Términos con más presencia (documentos que los contienen) en una clase."""
    textos = df.loc[df["etiqueta"] == clase, "texto"]
    if len(textos) > muestra:
        textos = textos.sample(muestra, random_state=42)
    ex = ExtractorCaracteristicas()
    prep = [ex.preprocesar(t) for t in textos]
    cv = CountVectorizer(stop_words="english", binary=True, min_df=2, max_features=5000)
    X = cv.fit_transform(prep)
    presencia = np.asarray(X.sum(axis=0)).ravel() / X.shape[0]
    idx = presencia.argsort()[::-1][:n]
    vocab = cv.get_feature_names_out()
    return [(vocab[i], round(float(presencia[i]), 3)) for i in idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ruta", nargs="?", help="CSV a explorar (por defecto, el primero de data/raw/)")
    ap.add_argument("--figuras", default=str(RAIZ / "docs" / "figuras"))
    args = ap.parse_args()

    if args.ruta:
        ruta = Path(args.ruta)
    else:
        csvs = sorted(RUTA_DATOS_CRUDOS.glob("*.csv"))
        if not csvs:
            sys.exit(f"No hay CSV en {RUTA_DATOS_CRUDOS}. Descarga el dataset primero.")
        ruta = csvs[0]

    cd = ConjuntoDatos(ruta)
    df = cd.cargar_datos()
    m = cd.metricas

    print(f"\n=== Archivo: {ruta.name} ===")
    print("\n-- Limpieza --")
    for k, v in m.items():
        print(f"  {k:>22}: {v:,}")
    if m["duplicados_eliminados"] / max(m["filas_originales"], 1) > 0.05:
        print("  AVISO: más del 5 % de duplicados; revisar la fuente.")

    hay_urls = bool(df["tiene_urls"].sum() > 0)
    resumen = {"archivo": ruta.name, "limpieza": m, "clases": {}}
    print("\n-- Por clase --")
    for clase in (0, 1):
        sub = df[df["etiqueta"] == clase]
        largo = sub["texto"].str.len()
        palabras = sub["texto"].str.split().str.len()
        info = {
            "correos": int(len(sub)),
            "caracteres_media": round(float(largo.mean()), 1),
            "caracteres_mediana": float(largo.median()),
            "caracteres_p95": float(largo.quantile(0.95)),
            "palabras_mediana": float(palabras.median()),
            # None si el archivo no trae información de enlaces (evita un 0 % falso)
            "con_urls_pct": round(float(sub["tiene_urls"].mean() * 100), 1) if hay_urls else None,
            "terminos_frecuentes": terminos_frecuentes(df, clase),
        }
        resumen["clases"][NOMBRES[clase]] = info
        print(f"\n  [{NOMBRES[clase]}]  correos: {info['correos']:,}")
        print(f"    caracteres  media {info['caracteres_media']:,}  mediana {info['caracteres_mediana']:,.0f}  p95 {info['caracteres_p95']:,.0f}")
        print(f"    palabras    mediana {info['palabras_mediana']:,.0f}")
        if info["con_urls_pct"] is None:
            print("    con enlaces no disponible (el archivo no trae esa información)")
        else:
            print(f"    con enlaces {info['con_urls_pct']} %")
        print("    términos más presentes (fracción de correos):")
        print("      " + ", ".join(f"{t} ({p})" for t, p in info["terminos_frecuentes"]))

    print("\n-- Muestras para verificar el significado de las etiquetas --")
    for clase in (0, 1):
        print(f"  [{NOMBRES[clase]}]")
        for t in df.loc[df["etiqueta"] == clase, "texto"].sample(3, random_state=1):
            print("    -", " ".join(t.split())[:110])

    carpeta = Path(args.figuras)
    carpeta.mkdir(parents=True, exist_ok=True)
    figura_balance(df, carpeta / "balance_clases.png")
    figura_longitud(df, carpeta / "longitud_por_clase.png")
    RUTA_DATOS_PROCESADOS.mkdir(parents=True, exist_ok=True)
    salida = RUTA_DATOS_PROCESADOS / "reporte_exploratorio.json"
    salida.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFiguras en {carpeta}\nReporte en {salida}\n")


if __name__ == "__main__":
    main()
