
from apoio import iniciar
from can_laboratorio import can_processos

if __name__ == "__main__":

    args, saida = iniciar(
        __file__,
        ciclos=50
    )

    can_processos(
        saida,
        ciclos=args.ciclos,
        hz=args.hz
    )
