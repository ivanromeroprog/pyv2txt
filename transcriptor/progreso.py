"""Control de cancelación y notificación de progreso.

La GUI necesita poder cancelar desde otro hilo y mostrar una barra de avance;
la consola necesita imprimir el avance en la terminal. Ambas cosas se
resuelven con los mismos parámetros que devuelve `crear_observadores`.
"""

from __future__ import annotations

import threading
from typing import Callable

from .resultado import Segmento

ProgresoCallback = Callable[[float | None, str], None]
SegmentoCallback = Callable[[Segmento], None]


class TokenCancelamiento:
    """Bandera segura entre hilos para abortar una transcripción en curso.

    faster-whisper entrega los segmentos como generador perezoso, así que
    basta con revisar esta bandera en cada iteración para cortar sin dejar
    hilos bloqueados.
    """

    def __init__(self) -> None:
        self._evento = threading.Event()

    def cancelar(self) -> None:
        self._evento.set()

    def limpiar(self) -> None:
        """Deja el token listo para reutilizarlo en otra transcripción."""
        self._evento.clear()

    @property
    def cancelado(self) -> bool:
        return self._evento.is_set()

    def __bool__(self) -> bool:
        return self._evento.is_set()


def crear_observadores(
    on_segmento: SegmentoCallback | None = None,
    on_progreso: ProgresoCallback | None = None,
    token: TokenCancelamiento | None = None,
) -> tuple[SegmentoCallback | None, ProgresoCallback | None, TokenCancelamiento]:
    """Normaliza los observadores opcionales de la API pública."""
    return on_segmento, on_progreso, token if token is not None else TokenCancelamiento()
