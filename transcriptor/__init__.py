"""Librería de transcripción de audio con faster-whisper.

Contiene toda la lógica del proyecto sin dependencias de interfaz, para que
la consola (`consola.py`) y la GUI (`interfaz.py`) puedan reutilizarla.

Uso mínimo:

    from transcriptor import Transcriptor, OpcionesTranscripcion

    t = Transcriptor(modelo="small", dispositivo="auto")
    resultado = t.transcribir("a.m4a", OpcionesTranscripcion(idioma="es"))
    print(resultado.texto)
"""

from .config import (
    COMPUTE_TYPE_POR_DISPOSITIVO,
    EXTENSIONES_AUDIO,
    FORMATOS_SALIDA,
    IDIOMA_AUTODETECTAR,
    IDIOMAS,
    MODELO_POR_DEFECTO,
    MODELOS_DISPONIBLES,
    etiqueta_idioma,
)
from .errores import (
    ArchivoInvalidoError,
    DependenciaFaltanteError,
    ModeloNoDisponibleError,
    TranscripcionCancelada,
    TranscriptorError,
)
from .exportadores import EXPORTADORES, MIME_SALIDA, escribir, serializar
from .motor import Transcriptor, detectar_disponibilidad_cuda, dispositivo_disponible
from .progreso import TokenCancelamiento
from .resultado import OpcionesTranscripcion, ResultadoTranscripcion, Segmento

__version__ = "1.0.0"

__all__ = [
    "ArchivoInvalidoError",
    "COMPUTE_TYPE_POR_DISPOSITIVO",
    "DependenciaFaltanteError",
    "EXPORTADORES",
    "EXTENSIONES_AUDIO",
    "FORMATOS_SALIDA",
    "IDIOMAS",
    "IDIOMA_AUTODETECTAR",
    "MIME_SALIDA",
    "MODELOS_DISPONIBLES",
    "MODELO_POR_DEFECTO",
    "ModeloNoDisponibleError",
    "OpcionesTranscripcion",
    "ResultadoTranscripcion",
    "Segmento",
    "TokenCancelamiento",
    "TranscripcionCancelada",
    "Transcriptor",
    "TranscriptorError",
    "__version__",
    "detectar_disponibilidad_cuda",
    "dispositivo_disponible",
    "escribir",
    "etiqueta_idioma",
    "serializar",
]
