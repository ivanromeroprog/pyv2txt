#!/usr/bin/env python3
"""Interfaz de línea de comandos para transcribir audio con faster-whisper.

Mantiene la compatibilidad con el script original `v2txt.py`:

    python consola.py entrada.m4a salida.txt

y suma opciones de modelo, dispositivo, idioma, formato de salida y progreso.

Todo el trabajo pesado vive en el paquete `transcriptor`; este archivo solo
traduce argumentos de la terminal a llamadas de la librería.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from transcriptor import (
    EXPORTADORES,
    IDIOMAS,
    MODELO_POR_DEFECTO,
    MODELOS_DISPONIBLES,
    ArchivoInvalidoError,
    ModeloNoDisponibleError,
    OpcionesTranscripcion,
    Transcriptor,
    TranscriptorError,
    detectar_disponibilidad_cuda,
    escribir,
    etiqueta_idioma,
)


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="consola.py",
        description="Transcribe un archivo de audio usando Whisper (faster-whisper).",
        epilog=(
            "Ejemplos:\n"
            "  python consola.py a.m4a salida.txt\n"
            "  python consola.py a.m4a -o salida.srt --formato srt\n"
            "  python consola.py a.m4a -o salida.txt --modelo medium --idioma en\n"
            "  python consola.py a.m4a -o salida.txt --dispositivo cpu\n"
            "  python consola.py --dispositivos\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("entrada", nargs="?", help="Archivo de audio o video de entrada")
    parser.add_argument("salida", nargs="?", help="Archivo de texto de salida (opcional)")

    parser.add_argument(
        "-o", "--salida-alternativa", dest="salida_alt", metavar="RUTA",
        help="Ruta de salida (equivalente al argumento posicional 'salida')",
    )
    parser.add_argument(
        "-m", "--modelo", default=MODELO_POR_DEFECTO,
        choices=MODELOS_DISPONIBLES,
        help="Tamaño del modelo Whisper (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dispositivo", default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Dónde ejecutar: auto intenta CUDA y cae a CPU (default: %(default)s)",
    )
    parser.add_argument(
        "-l", "--idioma", default="es",
        help=f"Código ISO del idioma, 'auto' para detectar. Default: es. "
             f"Comunes: {', '.join(IDIOMAS)}",
    )
    parser.add_argument(
        "-f", "--formato", default=None,
        choices=sorted(EXPORTADORES),
        help="Formato de salida; por defecto se deduce de la extensión. Default: deduce",
    )
    parser.add_argument(
        "--palabras", action="store_true",
        help="Incluir marcas de tiempo por palabra (más lento, útil para subtítulos)",
    )
    parser.add_argument(
        "--sin-vad", dest="vad", action="store_false",
        help="Desactivar el filtro de silencios (VAD)",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Texto inicial que guía el vocabulario (nombres propios, jerga)",
    )
    parser.add_argument(
        "--traducir", action="store_true",
        help="Traducir al inglés en vez de transcribir (tarea 'translate')",
    )
    parser.add_argument(
        "--silencioso", action="store_true",
        help="No mostrar el texto de los segmentos por pantalla",
    )
    parser.add_argument(
        "--dispositivos", action="store_true",
        help="Mostrar la información del sistema y salir",
    )
    return parser


def ruta_salida_por_defecto(entrada: str, formato: str) -> str:
    """Deduce la ruta de salida a partir del nombre de la entrada."""
    base = os.path.splitext(os.path.basename(entrada))[0]
    directorio = os.path.dirname(os.path.abspath(entrada))
    return os.path.join(directorio, f"{base}.{formato}")


def deducir_formato(salida: str) -> str:
    """Usa la extensión del archivo de salida; si no se reconoce, usa txt."""
    extension = os.path.splitext(salida)[1].lstrip(".").lower()
    return extension if extension in EXPORTADORES else "txt"


def mostrar_info_sistema() -> None:
    """Imprime el estado del entorno: Python, GPU y modelos sugeridos."""
    print(f"Python {sys.version.split()[0]} en {sys.platform}")
    disponible, motivo = detectar_disponibilidad_cuda()
    if disponible:
        print(f"GPU: disponible -> {motivo}. Se usará CUDA (float16).")
    else:
        print(f"GPU: no disponible -> {motivo}. Se usará CPU (int8).")
    print(f"Modelos: {', '.join(MODELOS_DISPONIBLES)} (default: {MODELO_POR_DEFECTO})")
    print(f"Formatos: {', '.join(sorted(EXPORTADORES))}")


def configurar_salida() -> None:
    """Fuerza UTF-8 en stdout/stderr para no romper acentos en la consola."""
    for flujo in (sys.stdout, sys.stderr):
        reconfigurable = getattr(flujo, "reconfigure", None)
        if reconfigurable:
            try:
                reconfigurable(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - terminal hostil
                pass


def main(argv: list[str] | None = None) -> int:
    configurar_salida()
    args = construir_parser().parse_args(argv)

    if args.dispositivos:
        mostrar_info_sistema()
        return 0

    if not args.entrada:
        construir_parser().error("falta el archivo de entrada (o usá --dispositivos)")

    salida = args.salida_alt or args.salida
    formato = args.formato or (deducir_formato(salida) if salida else "txt")
    if salida is None:
        salida = ruta_salida_por_defecto(args.entrada, formato)
    elif args.formato is None:
        # La extensión pedida manda sobre el sufijo real del archivo.
        salida = os.path.splitext(salida)[0] + f".{formato}"

    idioma = None if args.idioma.lower() in ("auto", "", "none") else args.idioma.lower()
    opciones = OpcionesTranscripcion(
        idioma=idioma,
        tarea="translate" if args.traducir else "transcribe",
        vad=args.vad,
        palabras=args.palabras,
        prompt_inicial=args.prompt,
    )

    inicio = time.perf_counter()

    def mostrar_estado(mensaje: str) -> None:
        if not args.silencioso:
            print(mensaje)

    def mostrar_segmento(segmento) -> None:
        if not args.silencioso:
            print(segmento.texto_limpio)

    def mostrar_progreso(fraccion: float | None, texto: str) -> None:
        if args.silencioso or not sys.stdout.isatty() or fraccion is None:
            return
        ancho = 24
        llenos = int(fraccion * ancho)
        barra = "#" * llenos + "." * (ancho - llenos)
        print(f"\r  [{barra}] {fraccion * 100:5.1f}%", end="", flush=True)

    transcriptor = Transcriptor(modelo=args.modelo, dispositivo=args.dispositivo)

    def mostrar_reinicio(mensaje: str) -> None:
        print(f"↻ {mensaje}")

    try:
        resultado = transcriptor.transcribir(
            args.entrada,
            opciones,
            on_segmento=mostrar_segmento,
            on_progreso=mostrar_progreso,
            on_estado=mostrar_estado,
            on_reinicio=mostrar_reinicio,
        )
    except ModeloNoDisponibleError as exc:
        print(f"\n✗ {exc}", file=sys.stderr)
        return 2
    except ArchivoInvalidoError as exc:
        print(f"\n✗ {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n✗ Cancelado por el usuario.", file=sys.stderr)
        return 130
    except TranscriptorError as exc:
        print(f"\n✗ Falló la transcripción: {exc}", file=sys.stderr)
        return 1

    if not sys.stdout.isatty() or args.silencioso:
        pass
    else:
        print()  # cierra la línea de la barra de progreso

    escribir(resultado, salida, formato)
    transcurrido = time.perf_counter() - inicio

    print(f"\n✓ Transcripción guardada en: {salida}")
    print(f"  Formato: {formato} | Idioma: {etiqueta_idioma(resultado.idioma)}")
    print(f"  Segmentos: {len(resultado.segmentos)} | Tiempo: {transcurrido:.1f}s")
    if resultado.vacio:
        print("  ⚠ No se detectó texto en el audio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
