"""Pruebas de B.2/B.3 con datos sintéticos (el dataset real es demasiado grande
para el repositorio). El CSV sintético imita el esquema de CEAS_08."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishing_detector.caracteristicas import ExtractorCaracteristicas  # noqa: E402
from phishing_detector.config import ConfigModelo  # noqa: E402
from phishing_detector.correo import CorreoElectronico  # noqa: E402
from phishing_detector.data import ConjuntoDatos  # noqa: E402


def _csv_ceas(tmp_path, filas):
    ruta = tmp_path / "ceas.csv"
    pd.DataFrame(
        filas, columns=["sender", "receiver", "date", "subject", "body", "label", "urls"]
    ).to_csv(ruta, index=False)
    return ruta


def _filas_base(n=20):
    filas = []
    for i in range(n):
        filas.append(["a@x.com", "b@y.com", "2008", f"Meeting notes {i}", f"See you at lunch {i}", 0, 0])
        filas.append(["z@evil.biz", "b@y.com", "2008", f"Urgent {i}", f"Verify your account http://evil.biz/{i}", 1, 1])
    return filas


def test_carga_columnas_ceas(tmp_path):
    cd = ConjuntoDatos(_csv_ceas(tmp_path, _filas_base()))
    df = cd.cargar_datos()
    assert list(df.columns) == ["texto", "etiqueta", "remitente", "tiene_urls"]
    assert set(df["etiqueta"]) == {0, 1}
    assert cd.metricas["filas_finales"] == 40
    assert df["texto"].iloc[0].startswith("Meeting notes 0")  # asunto + cuerpo


def test_elimina_nulos_y_duplicados(tmp_path):
    filas = _filas_base(5)
    filas.append(list(filas[0]))                      # duplicado exacto
    filas.append(["a", "b", "d", None, None, 0, 0])   # texto vacío
    filas.append(["a", "b", "d", "algo", "algo", None, 0])  # sin etiqueta
    cd = ConjuntoDatos(_csv_ceas(tmp_path, filas))
    cd.cargar_datos()
    m = cd.metricas
    assert m["duplicados_eliminados"] == 1
    assert m["texto_vacio"] == 1
    assert m["sin_etiqueta"] == 1
    assert m["filas_finales"] == 10


def test_texto_con_etiquetas_contradictorias_se_descarta(tmp_path):
    filas = _filas_base(5)
    filas.append(["a", "b", "d", "ambiguo", "mismo texto", 0, 0])
    filas.append(["a", "b", "d", "ambiguo", "mismo texto", 1, 0])
    cd = ConjuntoDatos(_csv_ceas(tmp_path, filas))
    df = cd.cargar_datos()
    assert cd.metricas["etiqueta_contradictoria"] == 2
    assert not df["texto"].str.contains("ambiguo").any()
    assert cd.metricas["filas_finales"] == 10


def test_etiqueta_desconocida_falla(tmp_path):
    filas = _filas_base(3)
    filas.append(["a", "b", "d", "x", "y", "quizas", 0])
    with pytest.raises(ValueError, match="no reconocidas"):
        ConjuntoDatos(_csv_ceas(tmp_path, filas)).cargar_datos()


def test_etiquetas_textuales(tmp_path):
    ruta = tmp_path / "kaggle.csv"
    pd.DataFrame(
        {"Email Text": ["hola amigo", "click here now"], "Email Type": ["Safe Email", "Phishing Email"]}
    ).to_csv(ruta, index=False)
    df = ConjuntoDatos(ruta).cargar_datos()
    assert df["etiqueta"].tolist() == [0, 1]


def test_columnas_text_combined_label(tmp_path):
    ruta = tmp_path / "phishing_email.csv"
    pd.DataFrame({"text_combined": ["hola amigo", "click here now"], "label": [0, 1]}).to_csv(ruta, index=False)
    df = ConjuntoDatos(ruta).cargar_datos()
    assert df["etiqueta"].tolist() == [0, 1]
    assert df["texto"].tolist() == ["hola amigo", "click here now"]


def test_division_estratificada_reproducible(tmp_path):
    cd = ConjuntoDatos(_csv_ceas(tmp_path, _filas_base(50)))
    X_tr, X_te, y_tr, y_te = cd.dividir_train_test(ConfigModelo())
    assert len(X_tr) == 80 and len(X_te) == 20
    assert abs(y_te.mean() - 0.5) < 0.05
    X_tr2, *_ = ConjuntoDatos(_csv_ceas(tmp_path, _filas_base(50))).dividir_train_test(ConfigModelo())
    assert X_tr == X_tr2


def test_preprocesar_reemplaza_url_email_numero():
    ex = ExtractorCaracteristicas()
    out = ex.preprocesar("Click <b>HERE</b>: http://evil.biz/login?id=7 or mail me@evil.biz, win $1,000!")
    assert "urltoken" in out and "emailtoken" in out and "numtoken" in out
    assert "http" not in out and "<b>" not in out and "evil" not in out


def test_extractor_requiere_ajustar_y_vectoriza():
    ex = ExtractorCaracteristicas(ConfigModelo(min_df=1))
    with pytest.raises(RuntimeError):
        ex.vectorizar(["hola"])
    textos = ["verify your account now http://a.biz", "lunch meeting tomorrow at noon"]
    ex.ajustar(textos)
    X = ex.vectorizar(textos)
    assert X.shape[0] == 2 and X.shape[1] == len(ex.vocabulario)
    assert "urltoken" in ex.vocabulario


def test_desde_texto_plano_cabeceras_y_enlaces():
    txt = "From: Banco <aviso@banco-falso.biz>\nSubject: Verifica tu cuenta\n\nEntra a http://banco-falso.biz/login. Gracias"
    c = CorreoElectronico.desde_texto_plano(txt)
    assert "aviso@banco-falso.biz" in c.remitente
    assert c.asunto == "Verifica tu cuenta"
    assert c.enlaces == ["http://banco-falso.biz/login"]
    assert c.cuerpo.startswith("Entra a")


def test_desde_texto_plano_sin_cabeceras():
    c = CorreoElectronico.desde_texto_plano("Hola, mira www.ejemplo.com")
    assert c.asunto == "" and c.remitente == ""
    assert c.enlaces == ["www.ejemplo.com"]
