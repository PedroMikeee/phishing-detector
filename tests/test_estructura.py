"""Prueba mínima de la fase de diseño: los módulos importan y la regla de
umbrales de ConfigOrquestador es coherente."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_modulos_importan():
    from phishing_detector import alerta, caracteristicas, config, correo, data, modelo, orquestador  # noqa: F401


def test_umbrales_ordenados():
    from phishing_detector.config import ConfigOrquestador

    cfg = ConfigOrquestador()
    assert 0.0 < cfg.umbral_medio < cfg.umbral_alto < 1.0


def test_correo_contenido():
    from phishing_detector.correo import CorreoElectronico

    c = CorreoElectronico(asunto="Urgente", cuerpo="Verifica tu cuenta")
    assert c.obtener_contenido() == "Urgente\nVerifica tu cuenta"
