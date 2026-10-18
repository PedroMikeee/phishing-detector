"""Clase CorreoElectronico (diagrama de clases, Fig. 1 del anteproyecto)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# URLs con esquema o que empiezan por www.
PATRON_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)
_PATRON_CABECERA = re.compile(r"^(from|de|subject|asunto)\s*:\s*(.*)$", re.IGNORECASE)


@dataclass
class CorreoElectronico:
    """Representa un correo de entrada al mecanismo.

    Atributos del diagrama: remitente, asunto, cuerpo, enlaces.
    """

    remitente: str = ""
    asunto: str = ""
    cuerpo: str = ""
    enlaces: list[str] = field(default_factory=list)

    def obtener_contenido(self) -> str:
        """Texto único que consume el extractor: asunto + cuerpo."""
        return f"{self.asunto}\n{self.cuerpo}".strip()

    @classmethod
    def desde_texto_plano(cls, texto: str) -> "CorreoElectronico":
        """Crea un correo a partir de texto pegado por el usuario (Streamlit).

        Si las primeras líneas son cabeceras `From:`/`Subject:` (o `De:`/
        `Asunto:`), las separa; si no, todo el texto es el cuerpo. Los
        enlaces se extraen con una expresión regular.
        """
        remitente, asunto = "", ""
        lineas = texto.splitlines()
        i = 0
        while i < len(lineas):
            m = _PATRON_CABECERA.match(lineas[i].strip())
            if not m:
                break
            clave = m.group(1).lower()
            if clave in ("from", "de"):
                remitente = m.group(2).strip()
            else:
                asunto = m.group(2).strip()
            i += 1
        cuerpo = "\n".join(lineas[i:]).strip()
        enlaces = [u.rstrip(".,;:") for u in PATRON_URL.findall(texto)]
        return cls(remitente=remitente, asunto=asunto, cuerpo=cuerpo, enlaces=enlaces)
