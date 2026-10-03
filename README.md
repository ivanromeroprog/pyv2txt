# Transcriptor de audio (Whisper)

Transcripción **local** de audio y video con
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (implementación
optimizada de Whisper sobre CTranslate2).

El proyecto tiene **dos interfaces separadas** que comparten la misma librería:

| Interfaz | Archivo | Para qué sirve |
| --- | --- | --- |
| Consola | `consola.py` | Automatizar, scripts, processing por lotes |
| Gráfica | `interfaz.py` | Uso manual: elegir archivo y ver el texto |

Y la lógica compartida vive en el paquete **`transcriptor/`**, sin ninguna
dependencia de interfaz.

---

## 1. Requerimientos

### 1.1 Sistema operativo

| Sistema | Soporte | Notas |
| --- | --- | --- |
| Windows 10/11 | Sí | Probado en Windows 11 con Python 3.12 |
| Linux | Sí | Probado en Ubuntu/Debian; la GUI puede necesitar `python3-tk` |
| macOS | Sí | En Apple Silicon (`arm64`) se recomienda `compute_type=float32` en CPU |

### 1.2 Python

- **Python 3.9 o superior** (probado con 3.12.8).
- Agregar la carpeta del proyecto al `PYTHONPATH` **o** ejecutar los scripts
  desde la raíz del proyecto (los imports de `transcriptor` son directos).

### 1.3 Dependencias de Python

Instalación recomendada:

```bash
pip install -r requirements.txt
```

| Paquete | Versión mínima | Función |
| --- | --- | --- |
| `faster-whisper` | >=1.0.0 | Wrapper de Whisper (API y decodificación) |
| `ctranslate2` | >=4.0.0 | Motor de inferencia en CPU y GPU |
| `av` (PyAV) | >=11.0.0 | Decodificación de audio y video |
| `onnxruntime` | >=1.17.0 | Runtime ONNX, usado en la ruta de CPU |
| `tokenizers` | >=0.13.0 | Tokenización de Whisper |
| `huggingface-hub` | >=0.20.0 | Descarga y caché de los pesos |
| `numpy` | >=1.24.0 | Manejo de arrays |
| `tqdm` | >=4.66.0 | Barras de progreso de la descarga |

Versiones efectivamente instaladas y verificadas al probar el proyecto:

```
faster-whisper 1.2.1
ctranslate2    4.8.2
av             15.1.0
onnxruntime    1.30.0
tokenizers     0.23.2
huggingface-hub 1.33.0
numpy          2.1.3
tqdm           4.70.1
```

### 1.4 Interfaz gráfica (solo `interfaz.py`)

- **tkinter**: viene incluido con Python en Windows y macOS.
- En Linux hay que instalarlo aparte:

```bash
sudo apt install python3-tk     # Debian / Ubuntu
sudo dnf install python3-tkinter # Fedora
sudo pacman -S tk               # Arch
```

`interfaz.py` no necesita ningún otro paquete (ni PyQt, ni Tkintermodern).

### 1.5 GPU (opcional)

La GPU es **opcional**: sin ella el proyecto funciona igual en CPU.

Para usarla hacen falta dos cosas:

1. **Una GPU NVIDIA** con drivers recientes y Compute Capability ≥ 7.0
   (para Whisper se recomienda una GPU con al menos 4 GB de VRAM).
2. **CUDA 12 y cuDNN 9**, ya sea del instalador oficial de NVIDIA o mediante
   los paquetes de pip:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

> **Importante:** en muchas máquinas el driver NVIDIA está instalado y la GPU
> aparece en el gestor, pero **faltan las librerías de CUDA**. En ese caso la
> carga del modelo en GPU funciona y la inferencia revienta con
> `Library cublas64_12.dll is not found or cannot be loaded`.
> Por eso el proyecto **cae automáticamente a CPU** (ver sección 4).
> Verificá tu entorno con:
>
> ```bash
> python consola.py --dispositivos
> ```

### 1.6 Hardware mínimo (CPU)

- **RAM:** 2 GB libres para `tiny`/`base`, 4 GB para `small`, 8 GB+ para `medium`/`large`.
- **Disco:** ~150 MB (`small`), ~500 MB (`medium`), ~3 GB (`large-v3`) para la
  caché del modelo, que se descarga la primera vez.
- **Velocidad aproximada en CPU** ( Ryzen 5 / 8 hilos, `small`, int8):
  entre 3x y 5x más rápido que el tiempo real del audio.

---

## 2. Estructura del proyecto

```
.
├── README.md              # este archivo
├── requirements.txt       # dependencias de Python
├── pyproject.toml         # metadatos, lint y entry point
├── abrir.bat              # lanzador de la GUI (Windows)
├── abrir.sh               # lanzador de la GUI (Linux / macOS)
├── v2txt.py               # script original (se conserva como referencia)
│
├── consola.py             # INTERFAZ DE CONSOLA (argumentos CLI)
├── interfaz.py            # INTERFAZ GRÁFICA (tkinter)
│
├── transcriptor/          # LIBRERÍA compartida por ambas interfaces
│   ├── __init__.py        # API pública
│   ├── config.py          # modelos, idiomas, formatos, extensiones
│   ├── errores.py         # excepciones propias
│   ├── resultado.py       # dataclasses (Segmento, ResultadoTranscripcion…)
│   ├── progreso.py        # cancelación y callbacks de progreso
│   ├── exportadores.py    # txt / srt / vtt / json / párrafo
│   └── motor.py           # Transcriptor: carga del modelo e inferencia
│
├── a.m4a                  # audio de ejemplo
└── s.txt                  # transcripción de ejemplo
```

**Por qué una librería aparte:** la consola y la GUI comparten exactamente la
misma lógica. Separarla evita duplicar el manejo de modelos, el fallback a CPU,
la cancelación y los formatos de salida. Las interfaces solo traduce
argumentos o clics a llamadas de la librería.

---

## 3. Instalación

```bash
# 1) Dependencias
pip install -r requirements.txt

# 2) (opcional) Instalar el paquete como proyecto, para tener el comando
#    `transcribir` disponible en cualquier carpeta:
pip install -e .
```

### Abrir la interfaz gráfica

| Sistema | Cómo |
| --- | --- |
| Windows | Doble clic en **`abrir.bat`** |
| Linux / macOS | `./abrir.sh` (la primera vez: `chmod +x abrir.sh`) |

Los dos lanzadores hacen lo mismo: buscan un Python (primero un entorno
virtual `.venv` o `venv` de la carpeta, luego `py -3` / `python3` /
`python`), verifican que estén `tkinter` y `faster-whisper`, y abren
`interfaz.py`. Si falta algo, te dicen exactamente qué comando ejecutar en
lugar de mostrar un error en inglés.

---

## 4. Uso

### 4.1 Interfaz gráfica

```bash
python interfaz.py
```

Funciones de la ventana:

- **Examinar…** para elegir el audio/video; la ruta de salida se autocompleta.
- Barra de avance y **Cancelar** (la transcripción corre en un hilo aparte, la
  ventana nunca se congela).
- El texto aparece a medida que se decodifica, y se puede copiar o guardar.
- Menú **Archivo / Herramientas / Ayuda**, con "Estado del sistema" para ver si
  hay GPU disponible.

### 4.2 Interfaz de consola

Compatible con el script original:

```bash
python consola.py a.m4a salida.txt
```

Con opciones:

```bash
# Elegir modelo y dispositivo
python consola.py a.m4a -o salida.txt --modelo medium --dispositivo cpu

# Idioma y subtítulos
python consola.py a.m4a -o salida.srt --idioma es
python consola.py a.m4a -o salida.srt --formato srt --palabras

# JSON con marcas de tiempo
python consola.py a.m4a -o salida.json --formato json

# Traducir al inglés en vez de transcribir
python consola.py a.m4a -o salida.txt --traducir

# Vocabulario y nombres propios que el modelo debe reconocer
python consola.py a.m4a -o salida.txt --prompt "ISFD, APAPED, Tecnicatura"

# Diagnosticar el entorno
python consola.py --dispositivos
```

Opciones completas: `python consola.py --help`.

### 4.3 Como librería

```python
from transcriptor import Transcriptor, OpcionesTranscripcion, escribir

transcriptor = Transcriptor(modelo="small", dispositivo="auto")

resultado = transcriptor.transcribir(
    "a.m4a",
    OpcionesTranscripcion(idioma="es", vad=True),
    on_segmento=lambda s: print(f"[{s.inicio:6.2f}s] {s.texto_limpio}"),
    on_progreso=lambda pct, txt: print(f"{pct or 0:.0%}", end="\r"),
)

escribir(resultado, "salida.srt", "srt")
print(resultado.idioma, resultado.duracion, len(resultado.segmentos))
```

El modelo se carga una sola vez y se reutiliza para varios archivos:

```python
for audio in ("uno.m4a", "dos.mp3"):
    resultado = transcriptor.transcribir(audio)   # ya está en memoria
```

Para cancelar desde otro hilo:

```python
from transcriptor import TokenCancelamiento

token = TokenCancelamiento()
# ... en otro hilo: token.cancelar()
resultado = transcriptor.transcribir("largo.wav", token=token)
```

---

## 5. Opciones

| Opción | Consola | GUI | Default | Descripción |
| --- | --- | --- | --- | --- |
| Modelo | `-m`, `--modelo` | Modelo | `small` | `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |
| Dispositivo | `-d`, `--dispositivo` | Dispositivo | `auto` | `auto`, `cuda`, `cpu` |
| Idioma | `-l`, `--idioma` | Idioma | `es` | Código ISO o `auto` |
| Formato | `-f`, `--formato` | Formato | según extensión | `txt`, `srt`, `vtt`, `json` |
| Filtro de silencios | `--sin-vad` | Filtrar silencios | activado | Recorta silencios antes de transcribir |
| Tiempo por palabra | `--palabras` | Marcar tiempo por palabra | desactivado | Útil para subtítulos |
| Prompt | `--prompt` | Prompt | vacío | Vocabulario / nombres propios |
| Traducir | `--traducir` | Traducir al inglés | desactivado | En vez de transcribir |

### Formatos de salida

- **`txt`**: un segmento por línea (igual que la salida del script original).
- **`srt`**: subtítulos numerados `00:00:01,740 --> 00:00:31,740`.
- **`vtt`**: subtítulos WebVTT (mismo contenido, cabecera `WEBVTT` y puntos).
- **`json`**: idioma, duración y segmentos con inicio, fin y palabras.

Todos los archivos se escriben en **UTF-8**.

---

## 6. Fallback automático a CPU

Este proyecto mantiene el criterio del script original: **si CUDA falla, se
reintenta en CPU** en lugar de abortar.

```
Cargando modelo en GPU (CUDA, float16)...
Modelo 'small' listo en cuda (float16).
Transcribiendo a.m4a en cuda (float16)...
⚠ CUDA falló durante la transcripción (Library cublas64_12.dll is not found or cannot be loaded).
→ Reintentando en CPU (int8)...
Cargando modelo en CPU (int8)...
✓ Transcripción guardada en: salida.txt
```

Detalles del comportamiento:

- El reintento es **completo** (se descarta el intento fallido), así el
  resultado no queda mezclado.
- Las interfaces reciben un aviso `on_reinicio` y limpian lo ya mostrado, para
  no duplicar texto en pantalla.
- Una vez que CUDA falla, la instancia lo recuerda (`cuda_deshabilitado`) y va
  directo a CPU en los archivos siguientes, sin volver a cargar el modelo dos
  veces.
- Si pedís `--dispositivo cuda` explícitamente, no hay fallback: el error se
  muestra. Para desactivarlo del todo:
  `set TRANSCRIPTOR_SIN_CUDA=1`.

---

## 7. Formatos de audio admitidos

Todo lo que PyAV pueda decodificar: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`,
`.opus`, `.aac`, `.wma`, `.mp4`, `.mkv`, `.mov`, `.webm`, `.avi`, `.mpg`, `.mpeg`,
`.ts`.

---

## 8. Problemas frecuentes

**`Library cublas64_12.dll is not found or cannot be loaded`**
Faltan las librerías de CUDA. Si no querés instalarlas, usá `--dispositivo cpu`
o `set TRANSCRIPTOR_SIN_CUDA=1`. El proyecto cae a CPU por su cuenta, pero es
más limpio evitar el intento.

**El modelo tarda mucho la primera vez**
Se está descargando de Hugging Face. Se cachea en
`%USERPROFILE%\.cache\huggingface` (Windows) o `~/.cache/huggingface` (Linux) y
las siguientes veces no se vuelve a descargar.

**Es muy lento**
Probá `--modelo tiny` o `base`, o activá el filtro de silencios. En CPU,
`small` va a `int8` automáticamente.

**Error al detectar GPU pero funciona**
`python consola.py --dispositivos` informa el estado real.

**Nombres propios mal transcritos**
Usá `--prompt "APAPED, ISFD, Tecnicatura Superior en Desarrollo de Software"`.

**Interfaz gráfica no abre en Linux**
Falta `python3-tk` (ver sección 1.4).
