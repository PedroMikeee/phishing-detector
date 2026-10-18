"""Entrenamiento y evaluación del modelo (CRISP-DM B.4 y B.5).

Uso (desde la raíz del proyecto, con el entorno virtual activo):

    python scripts/entrenar.py                                   # usa data/raw/phishing_email.csv
    python scripts/entrenar.py --calibrar                        # añade calibración sigmoide
    python scripts/entrenar.py --externo data/raw/Phishing_validation_emails.csv

Entrena con el 80 % del dataset, evalúa con el 20 % restante y, si se indica
--externo, evalúa también en un dataset distinto (validación externa).
Guarda el modelo en models/, las métricas en docs/resultados/ y dos figuras
en docs/figuras/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

from phishing_detector.caracteristicas import ExtractorCaracteristicas  # noqa: E402
from phishing_detector.config import (  # noqa: E402
    RAIZ,
    RUTA_DATOS_CRUDOS,
    RUTA_MODELO_ENTRENADO,
    ConfigModelo,
    ConfigOrquestador,
)
from phishing_detector.data import ConjuntoDatos  # noqa: E402
from phishing_detector.evaluacion import evaluar_probabilidades  # noqa: E402
from phishing_detector.modelo import ModeloNaiveBayes  # noqa: E402

COLOR_AZUL = "#2a78d6"
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
CUADRICULA = "#e1e0d9"
MAPA_AZUL = LinearSegmentedColormap.from_list("azul", ["#cde2fb", "#256abf", "#0d366b"])


def imprimir_metricas(titulo: str, m: dict) -> None:
    cm = m["matriz_confusion"]
    print(f"\n=== {titulo} (n = {m['n']:,}) ===")
    print(f"  exactitud    {m['exactitud']:.4f}")
    print(f"  precisión    {m['precision']:.4f}   (de lo que marca malicioso, cuánto lo es)")
    print(f"  sensibilidad {m['sensibilidad']:.4f}   (de lo malicioso, cuánto detecta)")
    print(f"  F1           {m['f1']:.4f}")
    if m["auc_roc"] is not None:
        print(f"  AUC-ROC      {m['auc_roc']:.4f}")
    print(f"  MSE (prob.)  {m['mse_probabilidad']:.4f}   (puntaje de Brier; 0 es lo ideal)")
    print(f"  R² (prob.)   {m['r2_probabilidad']:.4f}   (1 es lo ideal; negativo = peor que la media)")
    print(f"  matriz: VN {cm['tn']:,}  FP {cm['fp']:,}  FN {cm['fn']:,}  VP {cm['tp']:,}")
    print("  reparto por nivel de riesgo (% de cada clase real):")
    for clase, niveles in m["niveles_riesgo_pct"].items():
        print(f"    {clase:>9}: bajo {niveles['bajo']:>6}%  medio {niveles['medio']:>6}%  alto {niveles['alto']:>6}%")


def _estilo(ax):
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c3c2b7")
    ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=9)


def figura_matriz(m: dict, ruta: Path) -> None:
    cm = m["matriz_confusion"]
    datos = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    pct = datos / datos.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.0), facecolor=SUPERFICIE)
    ax.imshow(pct, cmap=MAPA_AZUL, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{datos[i, j]:,}\n({pct[i, j]:.1%})", ha="center", va="center",
                    fontsize=10, color="white" if pct[i, j] > 0.5 else TINTA)
    ax.set_xticks([0, 1], ["Legítimo", "Malicioso"], color=TINTA)
    ax.set_yticks([0, 1], ["Legítimo", "Malicioso"], color=TINTA)
    ax.set_xlabel("Predicho", color=TINTA_SECUNDARIA, fontsize=9)
    ax.set_ylabel("Real", color=TINTA_SECUNDARIA, fontsize=9)
    ax.set_title("Matriz de confusión (conjunto de prueba)", color=TINTA, fontsize=11, loc="left")
    for lado in ax.spines.values():
        lado.set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, facecolor=SUPERFICIE)
    plt.close(fig)


def figura_calibracion(y, p, ruta: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.8), facecolor=SUPERFICIE)
    for ax in (ax1, ax2):
        _estilo(ax)
        ax.yaxis.grid(True, color=CUADRICULA, linewidth=0.8)
        ax.set_axisbelow(True)
    frac, medio = calibration_curve(y, p, n_bins=10, strategy="uniform")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="#898781", linewidth=1.2)
    ax1.text(0.52, 0.42, "calibración perfecta", color=TINTA_SECUNDARIA, fontsize=8, rotation=35)
    ax1.plot(medio, frac, marker="o", markersize=7, linewidth=2, color=COLOR_AZUL,
             markeredgecolor=SUPERFICIE, markeredgewidth=1.5)
    ax1.set_xlabel("Probabilidad estimada de malicioso", color=TINTA_SECUNDARIA, fontsize=9)
    ax1.set_ylabel("Fracción real de maliciosos", color=TINTA_SECUNDARIA, fontsize=9)
    ax1.set_title("Calibración de la probabilidad", color=TINTA, fontsize=11, loc="left")
    ax2.hist(p, bins=20, range=(0, 1), color=COLOR_AZUL)
    ax2.set_yscale("log")
    ax2.set_ylim(bottom=0.8)
    ax2.set_xlabel("Probabilidad estimada de malicioso", color=TINTA_SECUNDARIA, fontsize=9)
    ax2.set_ylabel("Número de correos (escala log)", color=TINTA_SECUNDARIA, fontsize=9)
    ax2.set_title("Distribución de las probabilidades", color=TINTA, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, facecolor=SUPERFICIE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ruta", nargs="?", default=str(RUTA_DATOS_CRUDOS / "phishing_email.csv"))
    ap.add_argument("--externo", help="CSV de validación externa (otro dataset)")
    ap.add_argument("--calibrar", action="store_true", help="calibrar probabilidades (sigmoide)")
    ap.add_argument("--salida", default=str(RUTA_MODELO_ENTRENADO), help="ruta del modelo guardado")
    ap.add_argument("--figuras", default=str(RAIZ / "docs" / "figuras"))
    ap.add_argument("--resultados", default=str(RAIZ / "docs" / "resultados"))
    args = ap.parse_args()

    cfg, cfg_orq = ConfigModelo(), ConfigOrquestador()
    t0 = time.time()

    cd = ConjuntoDatos(Path(args.ruta))
    X_tr, X_te, y_tr, y_te = cd.dividir_train_test(cfg)
    print(f"Datos: {cd.metricas['filas_finales']:,} correos limpios -> "
          f"{len(X_tr):,} entrenamiento / {len(X_te):,} prueba")

    print("Preprocesando y ajustando TF-IDF (puede tardar un par de minutos)...")
    ex = ExtractorCaracteristicas(cfg)
    ex.ajustar(X_tr)
    Xtr, Xte = ex.vectorizar(X_tr), ex.vectorizar(X_te)
    print(f"Vocabulario: {len(ex.vocabulario):,} términos ({time.time() - t0:.0f} s)")

    modelo = ModeloNaiveBayes(cfg)
    modelo.entrenar(Xtr, y_tr, calibrar=args.calibrar)
    p_te = modelo.predecir(Xte)
    metricas = {"configuracion": {"calibrado": args.calibrar, **cfg.__dict__},
                "datos": cd.metricas,
                "prueba": evaluar_probabilidades(y_te, p_te, cfg=cfg_orq)}
    imprimir_metricas("Conjunto de prueba (20 %)", metricas["prueba"])

    hacia_mal, hacia_leg = modelo.terminos_mas_discriminantes(ex.vocabulario, n=15)
    print("\nTérminos que más inclinan hacia MALICIOSO:")
    print("  " + ", ".join(t for t, _ in hacia_mal))
    print("Términos que más inclinan hacia LEGÍTIMO:")
    print("  " + ", ".join(t for t, _ in hacia_leg))
    print("(Revísalos: si ves fechas, nombres o rasgos de formato, el modelo aprende la fuente, no el contenido.)")
    metricas["terminos"] = {"malicioso": [t for t, _ in hacia_mal], "legitimo": [t for t, _ in hacia_leg]}

    if args.externo:
        cd_ext = ConjuntoDatos(Path(args.externo))
        df_ext = cd_ext.cargar_datos()
        p_ext = modelo.predecir(ex.vectorizar(df_ext["texto"].tolist()))
        metricas["externa"] = evaluar_probabilidades(df_ext["etiqueta"].to_numpy(), p_ext, cfg=cfg_orq)
        imprimir_metricas(f"Validación externa: {Path(args.externo).name}", metricas["externa"])

    figuras, resultados = Path(args.figuras), Path(args.resultados)
    figuras.mkdir(parents=True, exist_ok=True)
    resultados.mkdir(parents=True, exist_ok=True)
    figura_matriz(metricas["prueba"], figuras / "matriz_confusion.png")
    figura_calibracion(y_te, p_te, figuras / "calibracion.png")
    (resultados / "metricas_entrenamiento.json").write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"extractor": ex, "modelo": modelo, "config": cfg, "metricas": metricas}, salida)
    print(f"\nModelo guardado en {salida}\nFiguras en {figuras}\nMétricas en {resultados}\n"
          f"Tiempo total: {time.time() - t0:.0f} s\n")


if __name__ == "__main__":
    main()
