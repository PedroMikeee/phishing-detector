"""Clase ConjuntoDatos (diagrama de clases, Fig. 1 del anteproyecto)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    COLUMNAS_ASUNTO,
    COLUMNAS_CUERPO,
    COLUMNAS_ETIQUETA,
    COLUMNAS_REMITENTE,
    COLUMNAS_URLS,
    MAPA_ETIQUETAS,
    ConfigModelo,
)


def _buscar_columna(df: pd.DataFrame, candidatas: tuple[str, ...]) -> str | None:
    """Devuelve el nombre real de la primera columna que coincide (sin
    distinguir mayúsculas ni espacios sobrantes), o None."""
    por_normalizado = {str(c).strip().lower(): c for c in df.columns}
    for nombre in candidatas:
        if nombre in por_normalizado:
            return por_normalizado[nombre]
    return None


def _mapear_etiqueta(valor) -> int | None:
    if pd.isna(valor):
        return None
    clave = str(valor).strip().lower()
    # Los CSV a veces guardan 1 como 1.0
    if clave.endswith(".0"):
        clave = clave[:-2]
    return MAPA_ETIQUETAS.get(clave, "__desconocida__")


@dataclass
class ConjuntoDatos:
    """Carga, normaliza y divide el dataset etiquetado.

    Contrato de salida de `cargar_datos`: un DataFrame con las columnas
    `texto` (asunto + cuerpo), `etiqueta` (0 = legítimo, 1 = malicioso),
    `remitente` y `tiene_urls` (estas dos pueden venir vacías si el CSV no
    las trae). Sin nulos ni textos duplicados.

    Nota de honestidad: en CEAS_08 la clase 1 corresponde a correo no
    deseado/malicioso (spam), no exclusivamente a phishing. Ver
    docs/DISENO.md, sección 3.
    """

    ruta: Path
    correos_etiquetados: pd.DataFrame | None = None
    metricas: dict = field(default_factory=dict)

    def cargar_datos(self) -> pd.DataFrame:
        df = pd.read_csv(self.ruta)
        filas_originales = len(df)

        col_cuerpo = _buscar_columna(df, COLUMNAS_CUERPO)
        col_etiqueta = _buscar_columna(df, COLUMNAS_ETIQUETA)
        if col_cuerpo is None or col_etiqueta is None:
            raise ValueError(
                f"No se encontraron columnas de texto y etiqueta en {self.ruta}. "
                f"Columnas disponibles: {list(df.columns)}"
            )
        col_asunto = _buscar_columna(df, COLUMNAS_ASUNTO)
        col_remitente = _buscar_columna(df, COLUMNAS_REMITENTE)
        col_urls = _buscar_columna(df, COLUMNAS_URLS)

        # Etiquetas: fallar si hay valores que no sabemos interpretar.
        mapeadas = df[col_etiqueta].map(_mapear_etiqueta)
        desconocidas = df.loc[mapeadas == "__desconocida__", col_etiqueta].unique()
        if len(desconocidas) > 0:
            raise ValueError(
                f"Etiquetas no reconocidas en '{col_etiqueta}': {list(desconocidas)[:10]}. "
                "Agrégalas a MAPA_ETIQUETAS en config.py indicando si son 0 o 1."
            )

        asunto = df[col_asunto].fillna("").astype(str) if col_asunto else ""
        cuerpo = df[col_cuerpo].fillna("").astype(str)
        texto = (asunto + "\n" + cuerpo).str.strip() if col_asunto else cuerpo.str.strip()

        limpio = pd.DataFrame(
            {
                "texto": texto,
                "etiqueta": mapeadas,
                "remitente": df[col_remitente].fillna("").astype(str) if col_remitente else "",
                "tiene_urls": pd.to_numeric(df[col_urls], errors="coerce").fillna(0).astype(int)
                if col_urls else 0,
            }
        )

        n_sin_etiqueta = int(limpio["etiqueta"].isna().sum())
        limpio = limpio.dropna(subset=["etiqueta"])
        limpio["etiqueta"] = limpio["etiqueta"].astype(int)

        n_texto_vacio = int((limpio["texto"] == "").sum())
        limpio = limpio[limpio["texto"] != ""]

        # Un mismo texto con etiquetas distintas es ambiguo: se descarta entero
        # en lugar de quedarnos con una etiqueta arbitraria.
        etiquetas_por_texto = limpio.groupby("texto")["etiqueta"].transform("nunique")
        n_contradictorios = int((etiquetas_por_texto > 1).sum())
        limpio = limpio[etiquetas_por_texto == 1]

        n_duplicados = int(limpio.duplicated(subset=["texto"]).sum())
        limpio = limpio.drop_duplicates(subset=["texto"]).reset_index(drop=True)

        conteo = limpio["etiqueta"].value_counts().to_dict()
        self.metricas = {
            "filas_originales": filas_originales,
            "sin_etiqueta": n_sin_etiqueta,
            "texto_vacio": n_texto_vacio,
            "etiqueta_contradictoria": n_contradictorios,
            "duplicados_eliminados": n_duplicados,
            "filas_finales": len(limpio),
            "legitimos": int(conteo.get(0, 0)),
            "maliciosos": int(conteo.get(1, 0)),
        }
        self.correos_etiquetados = limpio
        return limpio

    def dividir_train_test(self, cfg: ConfigModelo = ConfigModelo()):
        """División estratificada y reproducible.

        Devuelve (X_train, X_test, y_train, y_test) con textos y etiquetas.
        """
        if self.correos_etiquetados is None:
            self.cargar_datos()
        df = self.correos_etiquetados
        return train_test_split(
            df["texto"].tolist(),
            df["etiqueta"].to_numpy(),
            test_size=cfg.proporcion_prueba,
            stratify=df["etiqueta"],
            random_state=cfg.semilla,
        )
