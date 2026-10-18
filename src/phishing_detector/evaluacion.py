"""Métricas de evaluación (CRISP-DM B.5).

Cubre las métricas de clasificación y las de la probabilidad estimada
(MSE y R², sección B.5.1 del anteproyecto). Nota: el MSE entre la
probabilidad y la etiqueta 0/1 es el puntaje de Brier.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .config import ConfigOrquestador


def nivel_de_riesgo(p: float, cfg: ConfigOrquestador = ConfigOrquestador()) -> str:
    """Misma regla que usará el AgenteOrquestador (docs/DISENO.md, sección 5)."""
    if p < cfg.umbral_medio:
        return "bajo"
    if p < cfg.umbral_alto:
        return "medio"
    return "alto"


def evaluar_probabilidades(y_real, proba, umbral_clasificacion: float = 0.5,
                           cfg: ConfigOrquestador = ConfigOrquestador()) -> dict:
    """Calcula todas las métricas a partir de etiquetas reales y P(phishing)."""
    y = np.asarray(y_real).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= umbral_clasificacion).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    # Cómo se reparten los correos de cada clase real entre los niveles de riesgo.
    niveles = {"legitimo": {"bajo": 0, "medio": 0, "alto": 0},
               "malicioso": {"bajo": 0, "medio": 0, "alto": 0}}
    for real, prob in zip(y, p):
        clave = "malicioso" if real == 1 else "legitimo"
        niveles[clave][nivel_de_riesgo(prob, cfg)] += 1
    niveles_pct = {
        clase: {n: round(100 * c / max(sum(cuentas.values()), 1), 2) for n, c in cuentas.items()}
        for clase, cuentas in niveles.items()
    }

    return {
        "n": int(len(y)),
        "exactitud": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "sensibilidad": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
        "mse_probabilidad": float(mean_squared_error(y, p)),
        "r2_probabilidad": float(r2_score(y, p)),
        "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "niveles_riesgo_pct": niveles_pct,
    }
