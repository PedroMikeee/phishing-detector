"""Mide si las reglas y el aviso de idioma mejoran al modelo solo (CRISP-DM B.5).

Uso (desde la raíz del proyecto, con el entorno virtual activo y un modelo entrenado):

    python scripts/evaluar_mecanismo.py
    python scripts/evaluar_mecanismo.py --externo data/raw/Phishing_validation_emails.csv

Compara dos sistemas sobre los mismos correos:
  - "solo modelo": el nivel sale de la probabilidad de Naive Bayes.
  - "mecanismo":   el nivel que ve el usuario (modelo + reglas + aviso de idioma).

Se evalúa en (1) el 20 % de prueba del dataset de entrenamiento, con la misma
división que usó `scripts/entrenar.py`, y (2) el dataset externo si existe.
Para cada sistema se mide el comportamiento con dos criterios de alerta:
"nivel medio o alto" y "solo alto".

Cómo leer los resultados (limitaciones de la medida):
  - El corpus de entrenamiento tiene las palabras vacías ("the", "and"...) ya eliminadas,
    y el detector de idioma se apoya justo en ellas: ahí concluiría "no es inglés" casi
    siempre y apagaría el modelo. Por eso el 20 % de prueba se evalúa SIN el filtro de
    idioma (y se imprime cuántos correos habría descartado). El filtro sí se mantiene
    en el dataset externo, que tiene frases normales.
  - El 20 % de prueba es independiente de las reglas, pero llegó preprocesado
    (sin enlaces ni remitente), así que las reglas de enlaces y remitente no
    pueden dispararse ahí: mide sobre todo si las reglas de texto añaden falsas
    alarmas. No demuestra que los enlaces ayuden.
  - El dataset externo NO es independiente: al escribir las reglas de texto se
    vieron seis de sus filas, y las reglas `premio` y `datos_sensibles`
    coinciden con ellas. Sus cifras son optimistas para el mecanismo y solo
    indican una tendencia; además son 100 correos únicos.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishing_detector.alerta import NivelRiesgo  # noqa: E402
from phishing_detector.config import RAIZ, RUTA_DATOS_CRUDOS, RUTA_MODELO_ENTRENADO  # noqa: E402
from phishing_detector.data import ConjuntoDatos  # noqa: E402
from phishing_detector.idioma import detectar_idioma, modelo_aplicable  # noqa: E402
from phishing_detector.orquestador import AgenteOrquestador  # noqa: E402

CRITERIOS = {"nivel medio o alto": 1, "solo nivel alto": 2}


def resumen(y: list[int], niveles: list[NivelRiesgo]) -> dict:
    """Sensibilidad, precisión y falsas alarmas con cada criterio, y reparto por nivel."""
    salida: dict = {"criterios": {}, "reparto_pct": {}}
    for nombre, minimo in CRITERIOS.items():
        pred = [n.orden >= minimo for n in niveles]
        tp = sum(p and v == 1 for p, v in zip(pred, y))
        fp = sum(p and v == 0 for p, v in zip(pred, y))
        fn = sum((not p) and v == 1 for p, v in zip(pred, y))
        tn = sum((not p) and v == 0 for p, v in zip(pred, y))
        salida["criterios"][nombre] = {
            "sensibilidad": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "falsas_alarmas": fp / (fp + tn) if fp + tn else None,   # % de legítimos que alarman
            "matriz": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }
    for etiqueta, clase in (("legitimo", 0), ("malicioso", 1)):
        grupo = [n for n, v in zip(niveles, y) if v == clase]
        salida["reparto_pct"][etiqueta] = {
            n.value: round(100 * sum(g is n for g in grupo) / max(len(grupo), 1), 2) for n in NivelRiesgo}
    return salida


def evaluar(agente: AgenteOrquestador, textos: list[str], y: list[int]) -> dict:
    niveles_modelo, niveles_final, sin_modelo = [], [], 0
    probas = agente.modelo.predecir(agente.extractor.vectorizar(textos))
    for texto, p in zip(textos, probas):
        niveles_modelo.append(agente.evaluar_riesgo(float(p)))
        alerta = agente.recibir_correo(texto)
        niveles_final.append(alerta.nivel)
        sin_modelo += alerta.probabilidad is None
    agente.modulo_alerta.registro.clear()   # no acumular miles de alertas en memoria
    return {"n": len(y), "no_ingles_sin_modelo": sin_modelo,
            "solo_modelo": resumen(y, niveles_modelo), "mecanismo": resumen(y, niveles_final)}


def imprimir(titulo: str, r: dict) -> None:
    print(f"\n=== {titulo} (n = {r['n']:,}; sin usar el modelo por idioma: {r['no_ingles_sin_modelo']:,}) ===")
    for criterio in CRITERIOS:
        print(f"  Alerta con {criterio}:")
        print(f"    {'':<22}{'solo modelo':>13}{'mecanismo':>13}")
        for clave, nombre in (("sensibilidad", "sensibilidad"), ("precision", "precisión"),
                              ("falsas_alarmas", "falsas alarmas")):
            a, b = (r[s]["criterios"][criterio][clave] for s in ("solo_modelo", "mecanismo"))
            fa = lambda v: "      n/d" if v is None else f"{100 * v:>8.1f} %"  # noqa: E731
            print(f"    {nombre:<22}{fa(a):>13}{fa(b):>13}")
    print("  Reparto por nivel (% de cada clase real):")
    for sistema in ("solo_modelo", "mecanismo"):
        for clase, niv in r[sistema]["reparto_pct"].items():
            print(f"    {sistema:<12} {clase:>9}: bajo {niv['bajo']:>6}%  medio {niv['medio']:>6}%  alto {niv['alto']:>6}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=str(RUTA_MODELO_ENTRENADO))
    ap.add_argument("--entrenamiento", default=str(RUTA_DATOS_CRUDOS / "phishing_email.csv"),
                    help="CSV con el que se entrenó: se evalúa su 20 %% de prueba")
    ap.add_argument("--externo", default=str(RUTA_DATOS_CRUDOS / "Phishing_validation_emails.csv"))
    ap.add_argument("--resultados", default=str(RAIZ / "docs" / "resultados"))
    args = ap.parse_args()

    try:
        agente = AgenteOrquestador.desde_archivo(Path(args.modelo))
    except FileNotFoundError as e:
        sys.exit(str(e))
    if getattr(agente.modelo, "ejemplos_agregados", 0):
        print(f"AVISO: el modelo se actualizó con {agente.modelo.ejemplos_agregados:,} ejemplos nuevos; "
              "el 20 % de prueba puede incluirlos y la medida no sería independiente.")

    resultados: dict = {}
    t0 = time.time()
    ruta_ent = Path(args.entrenamiento)
    if ruta_ent.exists():
        _, X_te, _, y_te = ConjuntoDatos(ruta_ent).dividir_train_test(agente.modelo.cfg)
        descartados = sum(not modelo_aplicable(detectar_idioma(t)) for t in X_te)
        agente_sin_idioma = AgenteOrquestador(
            agente.extractor, agente.modelo, cfg=replace(agente.cfg, verificar_idioma=False))
        resultados["prueba_20pct"] = evaluar(agente_sin_idioma, X_te, [int(v) for v in y_te])
        resultados["prueba_20pct"]["descartados_por_idioma_si_se_aplicara"] = descartados
        imprimir(f"Prueba 20 % de {ruta_ent.name} (sin filtro de idioma)", resultados["prueba_20pct"])
        print(f"  Nota: con el filtro de idioma activo se habrían descartado {descartados:,} de {len(X_te):,} "
              "correos; este corpus no tiene palabras vacías y el detector no es aplicable.")
    else:
        print(f"(No se evalúa el 20 % de prueba: no existe {ruta_ent})")

    ruta_ext = Path(args.externo)
    if ruta_ext.exists():
        df = ConjuntoDatos(ruta_ext).cargar_datos()
        resultados["externo"] = evaluar(agente, df["texto"].tolist(), [int(v) for v in df["etiqueta"]])
        imprimir(f"Validación externa: {ruta_ext.name}", resultados["externo"])
    else:
        print(f"(No se evalúa el externo: no existe {ruta_ext})")

    destino = Path(args.resultados)
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "evaluacion_mecanismo.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en {destino / 'evaluacion_mecanismo.json'} ({time.time() - t0:.0f} s)\n")


if __name__ == "__main__":
    main()
