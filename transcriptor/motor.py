"""Núcleo de transcripción: carga del modelo y ejecución de faster-whisper.

Este módulo no sabe nada de interfaces de usuario. Expone `Transcriptor` y
`dispositivo_disponible`, que usan tanto `consola.py` como `interfaz.py`.
"""

from __future__ import annotations

import os
from typing import Any

from .config import (
    COMPUTE_TYPE_POR_DISPOSITIVO,
    MODELO_POR_DEFECTO,
    MODELOS_DISPONIBLES,
)
from .errores import (
    ArchivoInvalidoError,
    DependenciaFaltanteError,
    ModeloNoDisponibleError,
    TranscripcionCancelada,
)
from .progreso import ProgresoCallback, SegmentoCallback, TokenCancelamiento
from .resultado import OpcionesTranscripcion, ResultadoTranscripcion, Segmento


def _importar_whisper() -> Any:
    """Importa faster-whisper con un mensaje de error útil si falta."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise DependenciaFaltanteError(
            "Falta 'faster-whisper'. Instalá las dependencias con:\n"
            "    pip install -r requirements.txt"
        ) from exc
    return WhisperModel


def _importar_ctranslate2() -> Any | None:
    """Importa ctranslate2 para poder contar GPUs (opcional)."""
    try:
        import ctranslate2
    except ImportError:  # pragma: no cover - depende del entorno
        return None
    return ctranslate2


def detectar_disponibilidad_cuda() -> tuple[bool, str]:
    """Indica si hay GPU utilizable. Devuelve `(disponible, motivo)`.

    El motivo sirve para explicarle al usuario por qué se cae a CPU en lugar
    de fallar en silencio.
    """
    ct2 = _importar_ctranslate2()
    if ct2 is None:
        return False, "ctranslate2 no está instalado"

    try:
        conteo = ct2.get_cuda_device_count()
    except Exception as exc:
        return False, f"ctranslate2 no detecta GPU ({exc})"

    if conteo < 1:
        return False, "no hay ninguna GPU CUDA disponible"
    return True, f"{conteo} GPU(s) CUDA detectada(s)"


def dispositivo_disponible(preferido: str = "auto") -> tuple[str, str]:
    """Resuelve `(preferido, "auto")` a un dispositivo concreto usable.

    Devuelve el par `(dispositivo, compute_type)`.
    """
    if preferido == "cuda":
        disponible, _ = detectar_disponibilidad_cuda()
        if not disponible:
            raise ModeloNoDisponibleError(
                "Se pidió CUDA pero no hay GPU disponible. Usá --dispositivo cpu."
            )
        return "cuda", COMPUTE_TYPE_POR_DISPOSITIVO["cuda"]

    if preferido == "cpu":
        return "cpu", COMPUTE_TYPE_POR_DISPOSITIVO["cpu"]

    if preferido != "auto":
        raise ModeloNoDisponibleError(f"Dispositivo desconocido: {preferido!r}")

    if os.environ.get("TRANSCRIPTOR_SIN_CUDA"):
        return "cpu", COMPUTE_TYPE_POR_DISPOSITIVO["cpu"]

    disponible, _ = detectar_disponibilidad_cuda()
    if disponible:
        return "cuda", COMPUTE_TYPE_POR_DISPOSITIVO["cuda"]
    return "cpu", COMPUTE_TYPE_POR_DISPOSITIVO["cpu"]


class Transcriptor:
    """Transcriptor de audio con faster-whisper, reutilizable entre llamadas.

    El modelo se carga una sola vez (`cargar()` es idempotente) y se puede
    reutilizar para varios archivos sin volver a pagar el coste de carga.

    ```python
    tr = Transcriptor(modelo="small", dispositivo="auto")
    tr.cargar()
    resultado = tr.transcribir("a.m4a", OpcionesTranscripcion(idioma="es"))
    ```
    """

    def __init__(
        self,
        modelo: str = MODELO_POR_DEFECTO,
        dispositivo: str = "auto",
        compute_type: str | None = None,
        download_root: str | None = None,
        cpu_threads: int = 0,
    ) -> None:
        if modelo not in MODELOS_DISPONIBLES:
            raise ModeloNoDisponibleError(
                f"Modelo desconocido: {modelo!r}. "
                f"Disponibles: {', '.join(MODELOS_DISPONIBLES)}"
            )
        self.modelo = modelo
        self.preferencia_dispositivo = dispositivo
        self.compute_type_pedido = compute_type
        self.download_root = download_root
        self.cpu_threads = cpu_threads

        self.dispositivo: str | None = None
        self.compute_type: str | None = None
        self.cuda_deshabilitado = False
        self._modelo_cargado: Any = None

    # ------------------------------------------------------------------ carga

    @property
    def cargado(self) -> bool:
        return self._modelo_cargado is not None

    def cargar(self, on_estado: Any = None) -> Transcriptor:
        """Carga el modelo si aún no está en memoria. Devuelve `self`.

        `on_estado` es un callback opcional `Callable[[str], None]` usado para
        informar el avance (lo que la GUI muestra como estado).
        """
        if self.cargado:
            return self

        WhisperModel = _importar_whisper()
        # Si ya sabemos que CUDA no sirve en esta máquina, no se vuelve a
        # intentar: se va directo a CPU para no cargar el modelo dos veces.
        preferencia = self.preferencia_dispositivo
        if preferencia == "auto" and self.cuda_deshabilitado:
            preferencia = "cpu"
        self.dispositivo, self.compute_type = dispositivo_disponible(preferencia)
        if self.compute_type_pedido:
            self.compute_type = self.compute_type_pedido

        def avisar(mensaje: str) -> None:
            if on_estado:
                on_estado(mensaje)

        avisos = {
            "cuda": "Cargando modelo en GPU (CUDA, float16)...",
            "cpu": "Cargando modelo en CPU (int8)...",
        }
        avisar(avisos[self.dispositivo])

        argumentos: dict[str, Any] = {
            "device": self.dispositivo,
            "compute_type": self.compute_type,
        }
        if self.download_root:
            argumentos["download_root"] = self.download_root
        if self.cpu_threads:
            argumentos["cpu_threads"] = self.cpu_threads

        try:
            self._modelo_cargado = WhisperModel(self.modelo, **argumentos)
        except Exception as exc:
            # Mismo criterio que el script original: si CUDA falla en modo
            # "auto", se reintenta en CPU en lugar de rendirse.
            if self.preferencia_dispositivo == "auto" and self.dispositivo == "cuda":
                avisar(f"⚠ CUDA falló al cargar ({exc}). Reintentando en CPU...")
                self.cuda_deshabilitado = True
                self.dispositivo = "cpu"
                self.compute_type = COMPUTE_TYPE_POR_DISPOSITIVO["cpu"]
                argumentos["device"] = self.dispositivo
                argumentos["compute_type"] = self.compute_type
                try:
                    self._modelo_cargado = WhisperModel(self.modelo, **argumentos)
                except Exception as exc_cpu:
                    raise ModeloNoDisponibleError(
                        f"No se pudo cargar el modelo {self.modelo!r} en "
                        f"{self.dispositivo} ni en cpu: {exc_cpu}"
                    ) from exc_cpu
            else:
                raise ModeloNoDisponibleError(
                    f"No se pudo cargar el modelo {self.modelo!r} en "
                    f"{self.dispositivo} ({self.compute_type}): {exc}"
                ) from exc

        avisar(f"Modelo {self.modelo!r} listo en {self.dispositivo} ({self.compute_type}).")
        return self

    def descargar(self) -> None:
        """Libera el modelo de memoria (útil antes de cerrar la GUI)."""
        self._modelo_cargado = None

    # ----------------------------------------------------------- transcripción

    def transcribir(
        self,
        entrada: str | os.PathLike[str],
        opciones: OpcionesTranscripcion | None = None,
        on_segmento: SegmentoCallback | None = None,
        on_progreso: ProgresoCallback | None = None,
        token: TokenCancelamiento | None = None,
        on_estado: Any = None,
        on_reinicio: Any = None,
    ) -> ResultadoTranscripcion:
        """Transcribe `entrada` y devuelve el `ResultadoTranscripcion`.

        Los callbacks son opcionales y se invocan así:
          - `on_estado(mensaje)`: mensajes de estado legibles.
          - `on_segmento(segmento)`: cada segmento a medida que se decodifica.
          - `on_progreso(0.0-1.0 o None, texto_segmento)`: avance del archivo.
          - `on_reinicio(mensaje)`: se dispara al reintentar en CPU, para que
            la interfaz descarte lo ya mostrado del intento fallido.
          - `token.cancelar()` desde otro hilo: aborta de forma segura.

        En modo `auto`, si CUDA falla durante la inferencia se reintenta
        entero en CPU, igual que hacía el script original `v2txt.py`.
        """
        opciones = opciones or OpcionesTranscripcion()
        ruta = self._validar_entrada(entrada)
        self.cargar(on_estado=on_estado)

        if token is not None:
            token.limpiar()

        try:
            return self._decodificar(
                ruta, opciones, on_segmento, on_progreso, token, on_estado
            )
        except TranscripcionCancelada:
            raise
        except Exception as exc:
            if not self._puede_reintentar_en_cpu(exc):
                raise ArchivoInvalidoError(
                    f"faster-whisper no pudo procesar {ruta}: {exc}"
                ) from exc

            # CUDA falló durante la inferencia: se descarta el intento y se
            # repite completo en CPU. Los segmentos ya emitidos se avisan para
            # que la interfaz no los duplique.
            self.cuda_deshabilitado = True
            motivo = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
            if on_estado:
                on_estado(f"⚠ CUDA falló durante la transcripción ({motivo}).")
                on_estado("→ Reintentando en CPU (int8)...")
            if on_reinicio:
                on_reinicio("Reintentando en CPU; se reinicia la transcripción.")
            if token is not None:
                token.limpiar()
            self.descargar()
            self.cargar(on_estado=on_estado)

        return self._decodificar(
            ruta, opciones, on_segmento, on_progreso, token, on_estado
        )

    def _decodificar(
        self,
        ruta: str,
        opciones: OpcionesTranscripcion,
        on_segmento: SegmentoCallback | None,
        on_progreso: ProgresoCallback | None,
        token: TokenCancelamiento | None,
        on_estado: Any,
    ) -> ResultadoTranscripcion:
        """Ejecuta la inferencia y devuelve el resultado. Sin reintentos."""
        vad_parameters = (
            {"min_silence_duration_ms": opciones.min_silencio_ms} if opciones.vad else None
        )

        argumentos: dict[str, Any] = {
            "task": opciones.tarea,
            "vad_filter": opciones.vad,
            "vad_parameters": vad_parameters,
            "word_timestamps": opciones.palabras,
            "temperature": opciones.temperatura,
            "beam_size": opciones.beam_size,
            "condition_on_previous_text": opciones.condicionar_anterior,
            "without_timestamps": opciones.sin_timestamps,
        }
        if opciones.idioma:
            argumentos["language"] = opciones.idioma
        if opciones.prompt_inicial:
            argumentos["initial_prompt"] = opciones.prompt_inicial

        if on_estado:
            on_estado(
                f"Transcribiendo {os.path.basename(ruta)} en "
                f"{self.dispositivo} ({self.compute_type})..."
            )

        generador, info = self._modelo_cargado.transcribe(ruta, **argumentos)

        duracion = float(getattr(info, "duration", 0.0) or 0.0)
        resultado = ResultadoTranscripcion(
            idioma=getattr(info, "language", None),
            probabilidad_idioma=float(getattr(info, "language_probability", 0.0) or 0.0),
            duracion=duracion,
        )

        for bruto in generador:
            if token is not None and token.cancelado:
                raise TranscripcionCancelada("Transcripción cancelada por el usuario.")

            texto = (bruto.text or "").strip()
            if not texto:
                continue

            segmento = Segmento(
                inicio=float(bruto.start),
                fin=float(bruto.end),
                texto=texto,
                palabras=tuple(
                    (float(p.start), float(p.end), p.word, float(p.probability))
                    for p in (getattr(bruto, "words", None) or ())
                ),
            )
            resultado.segmentos.append(segmento)

            if on_segmento:
                on_segmento(segmento)
            if on_progreso:
                fraccion = None
                if duracion > 0:
                    fraccion = max(0.0, min(1.0, segmento.fin / duracion))
                on_progreso(fraccion, texto)

        if token is not None and token.cancelado:
            raise TranscripcionCancelada("Transcripción cancelada por el usuario.")

        if on_estado:
            vacio = " (sin texto detectado)" if resultado.vacio else ""
            on_estado(
                f"Transcripción finalizada{vacio}: {len(resultado.segmentos)} segmento(s)."
            )

        return resultado

    def transcribir_a_archivo(
        self,
        entrada: str | os.PathLike[str],
        salida: str | os.PathLike[str],
        formato: str = "txt",
        **kwargs: Any,
    ) -> tuple[ResultadoTranscripcion, str]:
        """Atajo: transcribe y guarda. Devuelve `(resultado, ruta_salida)`."""
        from .exportadores import escribir

        resultado = self.transcribir(entrada, **kwargs)
        escribir(resultado, str(salida), formato)
        return resultado, str(salida)

    # ---------------------------------------------------------------- interno

    def _puede_reintentar_en_cpu(self, exc: Exception) -> bool:
        """Decide si un fallo merece un reintento en CPU.

        Solo cuando el dispositivo era `auto`+`cuda` y el error huele a CUDA.
        Una GPU puede estar "presente" para el driver y aun así faltar
        cuBLAS/cuDNN: ese caso es el que realmente cae a CPU.
        """
        if self.preferencia_dispositivo != "auto" or self.dispositivo != "cuda":
            return False

        marcadores = (
            "cuda", "cublas", "cudnn", "cudart", "gpu", "nvidia",
            "device", "driver", "cublaslt",
        )
        mensaje = f"{type(exc).__name__}: {exc}".lower()
        return any(marcador in mensaje for marcador in marcadores)

    @staticmethod
    def _validar_entrada(entrada: str | os.PathLike[str]) -> str:
        ruta = os.fspath(entrada)
        if not os.path.isfile(ruta):
            raise ArchivoInvalidoError(f"No existe el archivo: {ruta}")
        if os.path.getsize(ruta) == 0:
            raise ArchivoInvalidoError(f"El archivo está vacío: {ruta}")
        return ruta
