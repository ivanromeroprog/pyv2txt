"""Constantes y catálogos compartidos por la consola y la GUI."""

from __future__ import annotations

MODELO_POR_DEFECTO = "small"

MODELOS_DISPONIBLES: tuple[str, ...] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
)

#: compute_type recomendado por dispositivo (rápido y estable en cada uno).
COMPUTE_TYPE_POR_DISPOSITIVO: dict[str, str] = {
    "cuda": "float16",
    "cpu": "int8",
}

#: ``None`` = detección automática (el propio Whisper la hace).
IDIOMA_AUTODETECTAR: str | None = None

IDIOMAS: dict[str, str] = {
    "es": "Español",
    "en": "Inglés",
    "pt": "Portugués",
    "fr": "Francés",
    "de": "Alemán",
    "it": "Italiano",
    "ru": "Ruso",
    "ja": "Japonés",
    "zh": "Chino",
    "ar": "Árabe",
    "nl": "Neerlandés",
    "ca": "Catalán",
    "gl": "Gallego",
    "eu": "Euskera",
}

EXTENSIONES_AUDIO: tuple[str, ...] = (
    "*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg", "*.oga", "*.opus",
    "*.aac", "*.wma", "*.mp4", "*.mkv", "*.mov", "*.webm", "*.avi",
    "*.mpg", "*.mpeg", "*.ts",
)

FORMATOS_SALIDA: tuple[str, ...] = ("txt", "srt", "vtt", "json")


def etiqueta_idioma(codigo: str | None) -> str:
    """Nombre legible de un código ISO de idioma."""
    if not codigo:
        return "Auto (detectar)"
    return IDIOMAS.get(codigo, codigo)
