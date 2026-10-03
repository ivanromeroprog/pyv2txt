"""Estructuras de datos de entrada y salida de la transcripción.

Son dataclasses puras: no dependen de faster-whisper ni de tkinter, así que
la consola y la GUI pueden manipularlas sin problema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OpcionesTranscripcion:
    """Parámetros de una transcripción.

    Los valores por defecto replican el comportamiento de `v2txt.py`
    (español, sin VAD) más algunas mejoras que no cambian el texto final.
    """

    idioma: str | None = "es"
    tarea: str = "transcribe"
    vad: bool = True
    min_silencio_ms: int = 500
    palabras: bool = False
    prompt_inicial: str | None = None
    temperatura: float = 0.0
    beam_size: int = 5
    condicionar_anterior: bool = False
    sin_timestamps: bool = True

    @classmethod
    def desde_dict(cls, datos: dict[str, Any] | None) -> OpcionesTranscripcion:
        """Construye opciones ignorando claves desconocidas (de GUI o argparse)."""
        if not datos:
            return cls()
        campos = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in datos.items() if k in campos})


@dataclass(frozen=True)
class Segmento:
    """Un fragmento de audio ya transcrito."""

    inicio: float
    fin: float
    texto: str
    palabras: tuple[tuple[float, float, str, float], ...] = ()

    @property
    def duracion(self) -> float:
        return max(0.0, self.fin - self.inicio)

    @property
    def texto_limpio(self) -> str:
        return self.texto.strip()


@dataclass
class ResultadoTranscripcion:
    """Resultado completo de una transcripción."""

    segmentos: list[Segmento] = field(default_factory=list)
    idioma: str | None = None
    probabilidad_idioma: float = 0.0
    duracion: float = 0.0

    @property
    def texto(self) -> str:
        """Todo el texto unido con saltos de línea, sin líneas vacías."""
        return "\n".join(
            s.texto_limpio for s in self.segmentos if s.texto_limpio
        )

    @property
    def vacio(self) -> bool:
        return not any(s.texto_limpio for s in self.segmentos)
