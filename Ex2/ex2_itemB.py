
from apoio import iniciar, parametros_stream
from rede_laboratorio import comparar

if __name__ == "__main__":
    # Configuracao do experimento:
    # 320 frames para cada protocolo, a 30 FPS.
    # Resolucao: 640x480.
    # Compressao JPEG: qualidade 70.
    # UDP: datagramas de ate 1200 bytes.

    args, saida = iniciar(
        __file__,
        frames=320
    )

    configuracao = parametros_stream(args)

    resultados = comparar(
        saida,
        configuracao,
        ordem=("udp", "tcp")
    )

    print("\n=== COMPARACAO UDP x TCP ===")

    for protocolo in ("udp", "tcp"):
        r = resultados[protocolo]

        print(f"\nProtocolo: {protocolo.upper()}")
        print(f"Transmitidos: {r['transmitidos']}")
        print(f"Recebidos: {r['recebidos']}")
        print(f"Latencia media: {r['media_latencia_ms']:.3f} ms")
        print(f"Jitter: {r['jitter_std_ms']:.3f} ms")
        print(f"Frames perdidos: {r['frames_perdidos']}")
        print(f"Taxa de perda: {r['perda_frames_pct']:.2f}%")

    print("\nResultados salvos em:", saida)
