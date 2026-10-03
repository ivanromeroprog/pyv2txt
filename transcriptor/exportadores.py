"""Serialización de un `ResultadoTranscripcion` a texto plano.

Cada función devuelve un `str` listo para escribir en disco con UTF-8, para que
la consola y la GUI compartan exactamente el mismo formato de salida.
"""

from __future__ import annotations

from .resultado import ResultadoTranscripcion


def _sello(tiempo: float, separador: str) -> str:
    """Formatea segundos como HH:MM:SS<separador>mmm para SRT o VTT."""
    total_ms = max(0, int(round(tiempo * 1000)))
    h, resto = divmod(total_ms, 3_600_000)
    m, resto = divmod(resto, 60_000)
    s, ms = divmod(resto, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{separador}{ms:03d}"


def a_txt(resultado: ResultadoTranscripcion) -> str:
    """Un segmento por línea (idéntico a la salida original de `v2txt.py`)."""
    lineas = [s.texto_limpio for s in resultado.segmentos if s.texto_limpio]
    return "\n".join(lineas) + ("\n" if lineas else "")


def a_srt(resultado: ResultadoTranscripcion) -> str:
    """Subtítulos SRT numerados con marcas de tiempo."""
    bloques: list[str] = []
    indice = 0
    for segmento in resultado.segmentos:
        texto = segmento.texto_limpio
        if not texto:
            continue
        indice += 1
        bloques.append(
            f"{indice}\n"
            f"{_sello(segmento.inicio, ',')} --> {_sello(segmento.fin, ',')}\n"
            f"{texto}"
        )
    return "\n\n".join(bloques) + ("\n" if bloques else "")


def a_vtt(resultado: ResultadoTranscripcion) -> str:
    """Subtítulos WebVTT (mismo contenido que SRT, cabecera WEBVTT y puntos)."""
    bloques: list[str] = []
    for segmento in resultado.segmentos:
        texto = segmento.texto_limpio
        if not texto:
            continue
        bloques.append(
            f"{_sello(segmento.inicio, '.')} --> {_sello(segmento.fin, '.')}\n{texto}"
        )
    cuerpo = "\n\n".join(bloques)
    return "WEBVTT\n\n" + cuerpo + ("\n" if bloques else "")


def a_json(resultado: ResultadoTranscripcion) -> str:
    """Volcado legible por máquina con segmentos y marcas de tiempo."""
    import json

    datos = {
        "idioma": resultado.idioma,
        "probabilidad_idioma": round(resultado.probabilidad_idioma, 4),
        "duracion": round(resultado.duracion, 3),
        "segmentos": [
            {
                "inicio": round(s.inicio, 3),
                "fin": round(s.fin, 3),
                "texto": s.texto_limpio,
                "palabras": [
                    {"inicio": round(p[0], 3), "fin": round(p[1], 3), "palabra": p[2]}
                    for p in s.palabras
                ],
            }
            for s in resultado.segmentos
            if s.texto_limpio
        ],
    }
    return json.dumps(datos, ensure_ascii=False, indent=2) + "\n"


def a_parrafo(resultado: ResultadoTranscripcion) -> str:
    """Todo el texto como un único párrafo (cómodo para pegarlo en otro lado)."""
    partes = [s.texto_limpio for s in resultado.segmentos if s.texto_limpio]
    return " ".join(partes)


EXPORTADORES = {
    "txt": a_txt,
    "srt": a_srt,
    "vtt": a_vtt,
    "json": a_json,
    "parrafo": a_parrafo,
}

MIME_SALIDA = {
    "txt": "Archivos de texto (*.txt)",
    "srt": "Subtítulos SRT (*.srt)",
    "vtt": "Subtítulos WebVTT (*.vtt)",
    "json": "Archivo JSON (*.json)",
    "parrafo": "Archivo de texto (*.txt)",
}


def serializar(resultado: ResultadoTranscripcion, formato: str = "txt") -> str:
    """Serializa el resultado en el formato pedido.

    Lanza `ValueError` si el formato no existe, para que ambas interfaces
    puedan avisar del error sin duplicar la lista de formatos válidos.
    """
    try:
        exportador = EXPORTADORES[formato.lower().lstrip(".")]
    except KeyError:
        validos = ", ".join(sorted(EXPORTADORES))
        raise ValueError(f"Formato desconocido: {formato!r}. Disponibles: {validos}") from None
    return exportador(resultado)


def escribir(resultado: ResultadoTranscripcion, ruta: str, formato: str = "txt") -> str:
    """Serializa y guarda en `ruta`. Devuelve la ruta escrita."""
    with open(ruta, "w", encoding="utf-8", newline="\n") as archivo:
        archivo.write(serializar(resultado, formato))
    return ruta
