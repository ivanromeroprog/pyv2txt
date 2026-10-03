import argparse
from faster_whisper import WhisperModel


def cargar_modelo_cuda():
    print("Intentando cargar modelo en GPU (CUDA)...")

    modelo = WhisperModel(
        "small",
        device="cuda",
        compute_type="float16"
    )

    print("✓ Modelo CUDA cargado.")
    return modelo


def cargar_modelo_cpu():
    print("→ Cargando modelo en CPU...")

    modelo = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    print("✓ Modelo CPU cargado.")
    return modelo


def transcribir(modelo, entrada, salida):
    segments, info = modelo.transcribe(
        entrada,
        language="es"
    )

    with open(salida, "w", encoding="utf-8") as archivo:
        for segment in segments:
            texto = segment.text.strip()

            if texto:
                archivo.write(texto + "\n")
                print(texto)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe un archivo de audio usando Whisper."
    )

    parser.add_argument(
        "entrada",
        help="Archivo de audio de entrada"
    )

    parser.add_argument(
        "salida",
        help="Archivo de texto de salida"
    )

    args = parser.parse_args()

    # Primero intentamos CUDA
    try:
        modelo = cargar_modelo_cuda()

        print(f"\nTranscribiendo con GPU: {args.entrada}")

        transcribir(
            modelo,
            args.entrada,
            args.salida
        )

        print(f"\n✓ Transcripción guardada en: {args.salida}")
        return

    except Exception as e:
        print("\n⚠ CUDA falló durante la transcripción.")
        print(f"  Motivo: {e}")
        print("\n→ Intentando nuevamente utilizando CPU...\n")

    # Si CUDA falla, volvemos a empezar con CPU
    try:
        modelo = cargar_modelo_cpu()

        print(f"\nTranscribiendo con CPU: {args.entrada}")

        transcribir(
            modelo,
            args.entrada,
            args.salida
        )

        print(f"\n✓ Transcripción guardada en: {args.salida}")

    except Exception as e:
        print("\n✗ También falló la transcripción con CPU.")
        print(f"  Motivo: {e}")


if __name__ == "__main__":
    main()