#!/usr/bin/env python3
"""Interfaz gráfica para transcribir audio con faster-whisper.

Ventana de Tkinter con selección de archivo, opciones del modelo, barra de
avance, cancelación y vista previa del texto. La transcripción corre en un
hilo aparte para que la ventana siga respondiendo; los resultados vuelven al
hilo principal mediante una cola.

    python interfaz.py

Toda la lógica de transcripción viene del paquete `transcriptor`; este archivo
solo se ocupa de la presentación.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from transcriptor import (
    EXPORTADORES,
    EXTENSIONES_AUDIO,
    FORMATOS_SALIDA,
    IDIOMA_AUTODETECTAR,
    IDIOMAS,
    MIME_SALIDA,
    MODELO_POR_DEFECTO,
    MODELOS_DISPONIBLES,
    TokenCancelamiento,
    TranscripcionCancelada,
    Transcriptor,
    TranscriptorError,
    detectar_disponibilidad_cuda,
    escribir,
    etiqueta_idioma,
)
from transcriptor.resultado import OpcionesTranscripcion

TITULO = "Transcriptor de audio · Whisper"
TAMANO_VENTANA = "960x660"


class AppTranscriptor(tk.Tk):
    """Ventana principal de la aplicación."""

    def __init__(self) -> None:
        super().__init__()
        self.title(TITULO)
        self.geometry(TAMANO_VENTANA)
        self.minsize(720, 540)

        # --- estado interno -------------------------------------------------
        self._cola: queue.Queue[tuple[str, object]] = queue.Queue()
        self._token = TokenCancelamiento()
        self._hilo: threading.Thread | None = None
        self._transcriptor: Transcriptor | None = None
        self._resultado_actual = None
        self._ultima_salida: str | None = None

        self._configurar_estilo()
        self._crear_variables()
        self._construir_menu()
        self._construir_widgets()
        self._colocar_widgets()
        self._estado_inicial()
        self._enlazar_eventos()
        self.after(100, self._procesar_cola)

    # ------------------------------------------------------------- variables

    def _crear_variables(self) -> None:
        self.var_entrada = tk.StringVar()
        self.var_salida = tk.StringVar()
        self.var_modelo = tk.StringVar(value=MODELO_POR_DEFECTO)
        self.var_dispositivo = tk.StringVar(value="auto")
        self.var_idioma = tk.StringVar(value="es")
        self.var_formato = tk.StringVar(value="txt")
        self.var_prompt = tk.StringVar()
        self.var_vad = tk.BooleanVar(value=True)
        self.var_palabras = tk.BooleanVar(value=False)
        self.var_traducir = tk.BooleanVar(value=False)
        self.var_estado = tk.StringVar(value="Listo.")
        self.var_porcentaje = tk.DoubleVar(value=0.0)
        self.var_salvar_auto = tk.BooleanVar(value=True)

    # ---------------------------------------------------------------- estilo

    def _configurar_estilo(self) -> None:
        estilo = ttk.Style(self)
        for nombre in ("vista", "clam", "default"):
            if nombre in estilo.theme_names():
                estilo.theme_use(nombre)
                break
        estilo.configure("Titulo.TLabel", font=("Segoe UI", 15, "bold"))
        estilo.configure("Sub.TLabel", foreground="#555555")
        estilo.configure("Estado.TLabel", foreground="#333333")
        estilo.configure("Boton.TButton", padding=(10, 6))

    # ------------------------------------------------------------------ menu

    def _construir_menu(self) -> None:
        barra = tk.Menu(self)

        archivo = tk.Menu(barra, tearoff=0)
        archivo.add_command(label="Abrir audio…", accelerator="Ctrl+O",
                            command=self._elegir_entrada)
        archivo.add_command(label="Guardar transcripción como…", accelerator="Ctrl+S",
                            command=self._guardar_resultado)
        archivo.add_separator()
        archivo.add_command(label="Salir", command=self._al_cerrar)
        barra.add_cascade(label="Archivo", menu=archivo)

        opciones = tk.Menu(barra, tearoff=0)
        opciones.add_command(label="Abrir carpeta de salida",
                             command=lambda: self._abrir_carpeta())
        opciones.add_command(label="Limpiar registro", command=self._limpiar_log)
        barra.add_cascade(label="Herramientas", menu=opciones)

        ayuda = tk.Menu(barra, tearoff=0)
        ayuda.add_command(label="Estado del sistema", command=self._mostrar_estado_sistema)
        ayuda.add_command(label="Acerca de", command=self._acerca_de)
        barra.add_cascade(label="Ayuda", menu=ayuda)

        self.config(menu=barra)

    # --------------------------------------------------------------- widgets

    def _construir_widgets(self) -> None:
        self.lbl_titulo = ttk.Label(self, text="Transcriptor de audio", style="Titulo.TLabel")
        self.lbl_sub = ttk.Label(
            self,
            text="Transcripción local con faster-whisper (CUDA si hay GPU, si no CPU).",
            style="Sub.TLabel",
        )

        self.frm_archivo = ttk.LabelFrame(self, text="Archivos", padding=10)
        self.lbl_entrada = ttk.Label(self.frm_archivo, text="Audio:")
        self.entrada = ttk.Entry(self.frm_archivo, textvariable=self.var_entrada)
        self.btn_examinar = ttk.Button(self.frm_archivo, text="Examinar…",
                                      command=self._elegir_entrada)
        self.lbl_salida = ttk.Label(self.frm_archivo, text="Salida:")
        self.salida = ttk.Entry(self.frm_archivo, textvariable=self.var_salida)
        self.btn_guardar_como = ttk.Button(self.frm_archivo, text="Guardar como…",
                                           command=self._elegir_salida)
        self.chk_guardar_auto = ttk.Checkbutton(
            self.frm_archivo, text="Guardar automáticamente al terminar",
            variable=self.var_salvar_auto,
        )

        self.frm_opciones = ttk.LabelFrame(self, text="Opciones de transcripción", padding=10)
        # Cada fila va en su propio sub-frame. Si el prompt se pusiera en el
        # mismo grid con columnspan, Tk exigiría su ancho mínimo en todas las
        # columnas que abarca y los desplegables de arriba quedarían aplastados.
        self.frm_campos = ttk.Frame(self.frm_opciones)
        self.frm_checks = ttk.Frame(self.frm_opciones)
        self.frm_prompt = ttk.Frame(self.frm_opciones)

        self.lbl_modelo = ttk.Label(self.frm_campos, text="Modelo:")
        self.cmb_modelo = ttk.Combobox(
            self.frm_campos, textvariable=self.var_modelo,
            values=list(MODELOS_DISPONIBLES), state="readonly", width=10,
        )
        self.lbl_dispositivo = ttk.Label(self.frm_campos, text="Dispositivo:")
        self.cmb_dispositivo = ttk.Combobox(
            self.frm_campos, textvariable=self.var_dispositivo,
            values=("auto", "cuda", "cpu"), state="readonly", width=9,
        )
        self.lbl_idioma = ttk.Label(self.frm_campos, text="Idioma:")
        self._codigos_idioma: tuple[str | None, ...] = (IDIOMA_AUTODETECTAR, *IDIOMAS)
        self.cmb_idioma = ttk.Combobox(
            self.frm_campos, textvariable=self.var_idioma,
            values=[etiqueta_idioma(c) for c in self._codigos_idioma],
            # 20 caracteres: el más largo es "Auto (detectar)", que si se
            # recorta queda ilegible.
            state="readonly", width=20,
        )
        self.lbl_formato = ttk.Label(self.frm_campos, text="Formato:")
        self.cmb_formato = ttk.Combobox(
            self.frm_campos, textvariable=self.var_formato,
            values=list(FORMATOS_SALIDA), state="readonly", width=6,
        )
        self.chk_vad = ttk.Checkbutton(self.frm_checks, text="Filtrar silencios (VAD)",
                                       variable=self.var_vad)
        self.chk_palabras = ttk.Checkbutton(
            self.frm_checks, text="Marcar tiempo por palabra", variable=self.var_palabras
        )
        self.chk_traducir = ttk.Checkbutton(
            self.frm_checks, text="Traducir al inglés", variable=self.var_traducir
        )
        self.lbl_prompt = ttk.Label(self.frm_prompt, text="Prompt:")
        self.ent_prompt = ttk.Entry(self.frm_prompt, textvariable=self.var_prompt)

        self.frm_controles = ttk.Frame(self)
        self.btn_transcribir = ttk.Button(
            self.frm_controles, text="▶  Transcribir", style="Boton.TButton",
            command=self._iniciar_transcripcion,
        )
        self.btn_cancelar = ttk.Button(
            self.frm_controles, text="■  Cancelar", command=self._cancelar,
            state="disabled",
        )
        self.btn_copiar = ttk.Button(
            self.frm_controles, text="Copiar texto", command=self._copiar_texto,
            state="disabled",
        )
        self.btn_abrir_carpeta = ttk.Button(
            self.frm_controles, text="Abrir carpeta", command=self._abrir_carpeta
        )

        self.barra = ttk.Progressbar(self, variable=self.var_porcentaje, maximum=100.0)
        self.lbl_estado = ttk.Label(self, textvariable=self.var_estado, style="Estado.TLabel")

        self.frm_log = ttk.LabelFrame(self, text="Transcripción", padding=8)
        self.log = ScrolledText(
            self.frm_log, wrap="word", height=14, undo=True, font=("Consolas", 10)
        )
        self.barra_scroll = ttk.Scrollbar(self.frm_log, orient="vertical",
                                          command=self.log.yview)
        self.log.configure(yscrollcommand=self.barra_scroll.set)
        self.lbl_info = ttk.Label(self, text="", style="Sub.TLabel")

    def _colocar_widgets(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(7, weight=1)

        self.lbl_titulo.grid(row=0, column=0, sticky="w", padx=14, pady=(14, 0))
        self.lbl_sub.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 10))

        self.frm_archivo.grid(row=2, column=0, sticky="ew", padx=14)
        self.frm_archivo.columnconfigure(1, weight=1)
        self.lbl_entrada.grid(row=0, column=0, sticky="w", pady=3)
        self.entrada.grid(row=0, column=1, sticky="ew", padx=6, pady=3)
        self.btn_examinar.grid(row=0, column=2, pady=3)
        self.lbl_salida.grid(row=1, column=0, sticky="w", pady=3)
        self.salida.grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        self.btn_guardar_como.grid(row=1, column=2, pady=3)
        self.chk_guardar_auto.grid(row=2, column=1, sticky="w", padx=6, pady=(6, 0))

        self.frm_opciones.grid(row=3, column=0, sticky="ew", padx=14, pady=(10, 0))
        self.frm_opciones.columnconfigure(0, weight=1)

        # Fila 0: modelo, dispositivo, idioma y formato, a su ancho natural.
        self.frm_campos.grid(row=0, column=0, sticky="ew")
        for columna, (etiqueta, control) in enumerate((
            (self.lbl_modelo, self.cmb_modelo),
            (self.lbl_dispositivo, self.cmb_dispositivo),
            (self.lbl_idioma, self.cmb_idioma),
            (self.lbl_formato, self.cmb_formato),
        )):
            etiqueta.grid(row=0, column=columna * 2, sticky="w", padx=(0, 6), pady=3)
            control.grid(row=0, column=columna * 2 + 1, sticky="w", padx=(0, 16), pady=3)

        # Fila 1: las opciones booleanas, sin compartir celda con nada.
        self.frm_checks.grid(row=1, column=0, sticky="w", pady=(10, 0))
        for columna, check in enumerate(
            (self.chk_vad, self.chk_palabras, self.chk_traducir)
        ):
            check.grid(row=0, column=columna, sticky="w", padx=(0, 12))

        # Fila 2: el prompt se estira con la ventana.
        self.frm_prompt.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.frm_prompt.columnconfigure(1, weight=1)
        self.lbl_prompt.grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.ent_prompt.grid(row=0, column=1, sticky="ew")

        self.frm_controles.grid(row=4, column=0, sticky="ew", padx=14, pady=12)
        self.btn_transcribir.pack(side="left")
        self.btn_cancelar.pack(side="left", padx=8)
        self.btn_copiar.pack(side="left", padx=(0, 8))
        self.btn_abrir_carpeta.pack(side="left")

        self.barra.grid(row=5, column=0, sticky="ew", padx=14)
        self.lbl_estado.grid(row=6, column=0, sticky="w", padx=14, pady=(4, 0))

        self.frm_log.grid(row=7, column=0, sticky="nsew", padx=14, pady=(10, 0))
        self.frm_log.rowconfigure(0, weight=1)
        self.frm_log.columnconfigure(0, weight=1)
        self.log.grid(row=0, column=0, sticky="nsew")
        self.barra_scroll.grid(row=0, column=1, sticky="ns")

        self.lbl_info.grid(row=8, column=0, sticky="w", padx=14, pady=(8, 12))

    # ---------------------------------------------------------------- estado

    def _estado_inicial(self) -> None:
        disponible, motivo = detectar_disponibilidad_cuda()
        resumen_gpu = f"GPU disponible ({motivo})" if disponible else f"Sin GPU: {motivo}"
        self.lbl_info.configure(
            text=f"Entorno: {resumen_gpu}. Los modelos se descargan la primera vez."
        )
        self._escribir_log("Listo. Elegí un archivo de audio para empezar.")

    @property
    def en_curso(self) -> bool:
        return self._hilo is not None and self._hilo.is_alive()

    def _enlazar_eventos(self) -> None:
        self.bind("<Control-o>", lambda _e: self._elegir_entrada())
        self.bind("<Control-s>", lambda _e: self._guardar_resultado())
        self.var_dispositivo.trace_add("write", lambda *_: self._sugerir_dispositivo())
        self.protocol("WM_DELETE_WINDOW", self._al_cerrar)

    def _sugerir_dispositivo(self) -> None:
        """Si el usuario no tocó nada, informa qué dispositivo se usará."""
        if self.en_curso:
            return
        if self.var_dispositivo.get() == "auto":
            disponible, _ = detectar_disponibilidad_cuda()
            self.var_estado.set(
                "Listo. Se usará CUDA (float16)." if disponible
                else "Listo. Se usará CPU (int8)."
            )

    # ------------------------------------------------------------ selectores

    def _elegir_entrada(self) -> None:
        if self.en_curso:
            return
        patrones = " ".join(EXTENSIONES_AUDIO)
        tipos = [("Audio y video", patrones), ("Todos los archivos", "*.*")]
        ruta = filedialog.askopenfilename(
            title="Elegir archivo de audio", filetypes=tipos
        )
        if not ruta:
            return
        self.var_entrada.set(ruta)
        if not self.var_salida.get().strip():
            self._sugerir_salida(ruta)

    def _sugerir_salida(self, ruta_entrada: str) -> None:
        formato = self.var_formato.get()
        base = os.path.splitext(ruta_entrada)[0]
        directorio = os.path.dirname(os.path.abspath(ruta_entrada))
        self.var_salida.set(os.path.join(directorio, f"{os.path.basename(base)}.{formato}"))

    def _elegir_salida(self) -> None:
        formato = self.var_formato.get()
        tipo = MIME_SALIDA.get(formato, "Todos los archivos (*.*)")
        extension = formato if formato in ("txt", "srt", "vtt", "json") else "txt"
        sugerencia = self.var_salida.get() or f"transcripcion.{extension}"
        ruta = filedialog.asksaveasfilename(
            title="Guardar transcripción como",
            initialfile=os.path.basename(sugerencia),
            defaultextension=f".{extension}",
            filetypes=[(tipo, f"*.{extension}"), ("Todos los archivos", "*.*")],
        )
        if ruta:
            self.var_salida.set(ruta)
            extension_elegida = os.path.splitext(ruta)[1].lstrip(".").lower()
            if extension_elegida in EXPORTADORES:
                self.var_formato.set(extension_elegida)

    # --------------------------------------------------------------- acciones

    def _iniciar_transcripcion(self) -> None:
        if self.en_curso:
            return

        entrada = self.var_entrada.get().strip().strip('"')
        if not entrada:
            messagebox.showwarning(TITULO, "Elegí un archivo de audio de entrada.")
            return
        if not os.path.isfile(entrada):
            messagebox.showerror(TITULO, f"No existe el archivo:\n{entrada}")
            return

        formato = self.var_formato.get()
        salida = self.var_salida.get().strip().strip('"')
        if self.var_salvar_auto.get() and not salida:
            self._sugerir_salida(entrada)
            salida = self.var_salida.get().strip()

        opciones = OpcionesTranscripcion(
            idioma=self._codigo_idioma(),
            tarea="translate" if self.var_traducir.get() else "transcribe",
            vad=self.var_vad.get(),
            palabras=self.var_palabras.get(),
            prompt_inicial=self.var_prompt.get().strip() or None,
        )

        self.log.delete("1.0", "end")
        self._resultado_actual = None
        self._ultima_salida = None
        self.var_porcentaje.set(0.0)
        self._alternar_controles(activo=True)
        self._escribir_log(
            f"Iniciando: modelo={self.var_modelo.get()} | "
            f"dispositivo={self.var_dispositivo.get()} | "
            f"idioma={etiqueta_idioma(opciones.idioma)} | formato={formato}"
        )

        self._token.limpiar()
        self._hilo = threading.Thread(
            target=self._trabajo,
            args=(
                entrada, salida, formato, opciones,
                self.var_modelo.get(), self.var_dispositivo.get(),
                bool(self.var_salvar_auto.get()),
            ),
            daemon=True,
            name="transcripcion",
        )
        self._hilo.start()

    def _trabajo(
        self,
        entrada: str,
        salida: str,
        formato: str,
        opciones: OpcionesTranscripcion,
        modelo: str,
        dispositivo: str,
        guardar_auto: bool,
    ) -> None:
        """Hilo secundario: todo lo de aquí no puede tocar la interfaz.

        Los valores de Tkinter se reciben como argumentos desde el hilo
        principal: leer `StringVar` desde otro hilo no es seguro.
        """
        try:
            transcriptor = Transcriptor(modelo=modelo, dispositivo=dispositivo)
            self._transcriptor = transcriptor

            resultado = transcriptor.transcribir(
                entrada,
                opciones,
                on_segmento=self._encola_segmento,
                on_progreso=self._encola_progreso,
                on_estado=self._encola_estado,
                on_reinicio=self._encola_reinicio,
                token=self._token,
            )

            ruta_guardada = None
            if salida and guardar_auto:
                escribir(resultado, salida, formato)
                ruta_guardada = salida

            self._cola.put(("fin", (resultado, ruta_guardada)))

        except TranscripcionCancelada as exc:
            self._cola.put(("cancelado", str(exc)))
        except TranscriptorError as exc:
            self._cola.put(("error", str(exc)))
        except Exception as exc:  # noqa: BLE001 - la GUI no debe morir nunca
            self._cola.put(("error", f"Error inesperado: {exc}"))

    # Callbacks del hilo secundario: solo escriben en la cola.
    def _encola_estado(self, mensaje: str) -> None:
        self._cola.put(("estado", mensaje))

    def _encola_segmento(self, segmento) -> None:
        self._cola.put(("segmento", segmento.texto_limpio))

    def _encola_progreso(self, fraccion: float | None, _texto: str) -> None:
        self._cola.put(("progreso", fraccion))

    def _encola_reinicio(self, mensaje: str) -> None:
        self._cola.put(("reinicio", mensaje))

    # ------------------------------------------------- consumo desde el hilo UI

    def _procesar_cola(self) -> None:
        try:
            while True:
                tipo, datos = self._cola.get_nowait()
                if tipo == "estado":
                    self.var_estado.set(str(datos))
                elif tipo == "progreso":
                    self._actualizar_progreso(datos)
                elif tipo == "segmento":
                    self._agregar_linea(str(datos))
                elif tipo == "reinicio":
                    self._procesar_reinicio(str(datos))
                elif tipo == "fin":
                    self._procesar_fin(*datos)  # type: ignore[misc]
                elif tipo == "cancelado":
                    self._procesar_cancelado(str(datos))
                elif tipo == "error":
                    self._procesar_error(str(datos))
        except queue.Empty:
            pass
        self.after(100, self._procesar_cola)

    def _actualizar_progreso(self, fraccion: float | None) -> None:
        if fraccion is None:
            self.barra.config(mode="indeterminate")
            if str(self.barra.cget("mode")) != "indeterminate":
                self.barra.start(12)
        else:
            if str(self.barra.cget("mode")) == "indeterminate":
                self.barra.stop()
                self.barra.config(mode="determinate")
            self.var_porcentaje.set(fraccion * 100.0)

    def _procesar_reinicio(self, mensaje: str) -> None:
        """El motor reintenta en CPU: se borra el intento fallido."""
        self.log.delete("1.0", "end")
        self.var_porcentaje.set(0.0)
        self._resultado_actual = None
        self._ultima_salida = None
        self._escribir_log(f"↻ {mensaje}")

    def _procesar_fin(self, resultado, ruta_guardada: str | None) -> None:
        self._resultado_actual = resultado
        self._ultima_salida = ruta_guardada
        self.barra.stop()
        self.barra.config(mode="determinate")
        self.var_porcentaje.set(100.0)
        self._alternar_controles(activo=False)
        self.btn_copiar.config(state="normal")

        minutos = resultado.duracion / 60 if resultado.duracion else 0.0
        self.var_estado.set(
            f"Listo: {len(resultado.segmentos)} segmento(s) · "
            f"idioma {etiqueta_idioma(resultado.idioma)} · audio {minutos:.1f} min"
        )
        if ruta_guardada:
            self._escribir_log(f"\n✓ Guardado en: {ruta_guardada}")
            messagebox.showinfo(TITULO, f"Transcripción guardada en:\n{ruta_guardada}")
        elif resultado.vacio:
            messagebox.showwarning(TITULO, "No se detectó texto en el audio.")

    def _procesar_cancelado(self, mensaje: str) -> None:
        """El usuario canceló: no es un error, solo un estado final."""
        self.barra.stop()
        self.barra.config(mode="determinate")
        self._alternar_controles(activo=False)
        self.var_estado.set("Cancelado por el usuario.")
        self._escribir_log(f"\n■ {mensaje} El texto parcial quedó en pantalla.")
        # El texto parcial sirve para copiarlo a mano, aunque no se guardó.
        if self.log.get("1.0", "end").strip():
            self.btn_copiar.config(state="normal")

    def _procesar_error(self, mensaje: str) -> None:
        self.barra.stop()
        self.var_porcentaje.set(0.0)
        self._alternar_controles(activo=False)
        self.var_estado.set("Error durante la transcripción.")
        self._escribir_log(f"\n✗ {mensaje}")
        messagebox.showerror(TITULO, mensaje)

    # ------------------------------------------------------------- auxiliares

    def _alternar_controles(self, activo: bool) -> None:
        normal = "disabled" if activo else "normal"
        self.btn_transcribir.config(state=normal)
        self.btn_cancelar.config(state="normal" if activo else "disabled")
        for widget in (self.entrada, self.salida, self.ent_prompt, self.btn_examinar,
                       self.btn_guardar_como, self.cmb_modelo, self.cmb_dispositivo,
                       self.cmb_idioma, self.cmb_formato, self.chk_vad, self.chk_palabras,
                       self.chk_traducir, self.chk_guardar_auto):
            widget.config(state="disabled" if activo else ("readonly"
                               if isinstance(widget, ttk.Combobox) else "normal"))

    def _cancelar(self) -> None:
        if self.en_curso:
            self._token.cancelar()
            self.var_estado.set("Cancelando…")
            self.btn_cancelar.config(state="disabled")

    def _agregar_linea(self, texto: str) -> None:
        self.log.insert("end", texto + "\n")
        self.log.see("end")

    def _escribir_log(self, texto: str) -> None:
        self.log.insert("end", texto.rstrip("\n") + "\n")
        self.log.see("end")

    def _limpiar_log(self) -> None:
        if self.en_curso:
            return
        self.log.delete("1.0", "end")

    def _codigo_idioma(self) -> str | None:
        """Devuelve el código ISO del idioma elegido en el combobox.

        Se resuelve por índice y no por texto, para que el mapeo no dependa
        de cómo se vea la etiqueta ni de sus acentos.
        """
        indice = self.cmb_idioma.current()
        if indice < 0 or indice >= len(self._codigos_idioma):
            return None
        return self._codigos_idioma[indice]

    def _copiar_texto(self) -> None:
        texto = self.log.get("1.0", "end").strip()
        if not texto:
            return
        self.clipboard_clear()
        self.clipboard_append(texto)
        self.var_estado.set("Texto copiado al portapapeles.")
        self.btn_copiar.config(text="✓ Copiado", style="Boton.TButton")
        self.after(1500, lambda: self.btn_copiar.config(text="Copiar texto",
                                                        style="Boton.TButton"))

    def _guardar_resultado(self) -> None:
        if self.en_curso:
            return
        if self._resultado_actual is None:
            if not self.log.get("1.0", "end").strip():
                messagebox.showinfo(TITULO, "Todavía no hay nada que guardar.")
            else:
                ruta = filedialog.asksaveasfilename(
                    title="Guardar transcripción como", defaultextension=".txt",
                    filetypes=[(MIME_SALIDA[self.var_formato.get()], "*.txt"), ("Todos", "*.*")],
                )
                if ruta:
                    with open(ruta, "w", encoding="utf-8") as archivo:
                        archivo.write(self.log.get("1.0", "end").strip() + "\n")
                    self.var_estado.set(f"Guardado en: {ruta}")
            return
        if not self.var_salida.get().strip():
            self._elegir_salida()
        escribir(self._resultado_actual, self.var_salida.get().strip(),
                 self.var_formato.get())
        self.var_estado.set(f"Guardado en: {self.var_salida.get().strip()}")

    def _abrir_carpeta(self) -> None:
        ruta = self.var_salida.get().strip() or self.var_entrada.get().strip()
        carpeta = os.path.dirname(os.path.abspath(ruta)) if ruta else os.getcwd()
        if sys.platform.startswith("win"):
            os.startfile(carpeta)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", carpeta])
        else:
            subprocess.Popen(["xdg-open", carpeta])

    def _mostrar_estado_sistema(self) -> None:
        disponible, motivo = detectar_disponibilidad_cuda()
        modelo = self.var_modelo.get()
        messagebox.showinfo(
            TITULO,
            f"Python: {sys.version.split()[0]}\n"
            f"Plataforma: {sys.platform}\n"
            f"GPU: {'sí' if disponible else 'no'} ({motivo})\n"
            f"Modelo seleccionado: {modelo}\n"
            f"Formatos: {', '.join(FORMATOS_SALIDA)}",
        )

    def _acerca_de(self) -> None:
        from transcriptor import __version__

        messagebox.showinfo(
            TITULO,
            f"Transcriptor de audio v{__version__}\n\n"
            "Transcripción local con faster-whisper.\n"
            "Interfaz gráfica: interfaz.py\n"
            "Interfaz de consola: consola.py",
        )

    def _al_cerrar(self) -> None:
        if self.en_curso:
            if not messagebox.askokcancel(
                TITULO,
                "Hay una transcripción en curso.\n¿Cancelarla y salir?",
            ):
                return
            self._token.cancelar()
            self._hilo.join(timeout=5.0)
        if self._transcriptor is not None:
            self._transcriptor.descargar()
        self.destroy()


def main() -> int:
    app = AppTranscriptor()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
