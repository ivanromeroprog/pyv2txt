"""Excepciones propias del transcriptor.

Se separan del resto para que la consola y la GUI puedan mostrar mensajes
útiles al usuario sin depender de los textos internos de faster-whisper.
"""


class TranscriptorError(Exception):
    """Error base de la librería."""


class DependenciaFaltanteError(TranscriptorError):
    """faster-whisper (o una dependencia suya) no está instalada."""


class ModeloNoDisponibleError(TranscriptorError):
    """No se pudo cargar el modelo en el dispositivo pedido."""


class ArchivoInvalidoError(TranscriptorError):
    """El archivo de entrada no existe, está vacío o no es reproducible."""


class TranscripcionCancelada(TranscriptorError):
    """El usuario canceló la transcripción en curso."""
