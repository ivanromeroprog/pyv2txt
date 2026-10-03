#!/usr/bin/env bash
# =============================================================================
#  Lanzador de la interfaz grafica (Linux / macOS)
#
#  Uso:  ./abrir.sh
#
#  Busca Python (venv local -> python3 -> python), verifica las dependencias
#  y abre interfaz.py.
#
#  Si da permiso denegado:  chmod +x abrir.sh
# =============================================================================

set -u

cd "$(dirname "$0")" || {
    echo "[ERROR] No se pudo acceder a la carpeta del proyecto." >&2
    exit 1
}

echo
echo "  Iniciando el Transcriptor de audio..."
echo

# --- 1) Buscar un interprete de Python ---------------------------------------
PY=""
for candidate in ".venv/bin/python" "venv/bin/python"; do
    if [ -x "$candidate" ]; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PY="python3"
    elif command -v python >/dev/null 2>&1; then
        PY="python"
    fi
fi

if [ -z "$PY" ]; then
    cat >&2 <<'EOF'

  [ERROR] No se encontro Python en este equipo.

  En Debian/Ubuntu:  sudo apt install python3 python3-tk
  En Fedora:         sudo dnf install python3 python3-tkinter
  En Arch:           sudo pacman -S python tk
EOF
    exit 1
fi

# --- 2) Verificar dependencias ------------------------------------------------
echo "  Python: $PY"

if ! "$PY" -c "import tkinter" >/dev/null 2>&1; then
    cat >&2 <<'EOF'

  [ERROR] Falta tkinter, necesario para la interfaz grafica.

  En Debian/Ubuntu:  sudo apt install python3-tk
  En Fedora:         sudo dnf install python3-tkinter
  En Arch:           sudo pacman -S tk
EOF
    exit 1
fi

if ! "$PY" -c "import faster_whisper" >/dev/null 2>&1; then
    echo >&2
    echo "  [ERROR] Faltan las dependencias del proyecto." >&2
    echo >&2
    echo "  Ejecuta este comando y despues volve a abrir la interfaz:" >&2
    echo >&2
    echo "      $PY -m pip install -r requirements.txt" >&2
    exit 1
fi

# --- 3) Abrir la interfaz -----------------------------------------------------
exec "$PY" interfaz.py
