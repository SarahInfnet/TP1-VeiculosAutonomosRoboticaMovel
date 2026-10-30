"""Laboratório CAN: python-can com interface='virtual'.

Instalar python-can, NumPy e Matplotlib pelo requirements.txt do pacote.
O VirtualBus funciona dentro de um processo e não simula arbitragem ou bitrate.
Na atividade com dois processos, cada processo possui dois nós virtuais locais;
uma multiprocessing.Queue liga os gateways. Essa ponte IPC é explícita e não
representa um controlador/transceptor CAN físico. Ver documentação python-can.

CAN 2.0A: ID padrão de 11 bits, payload máximo 8 bytes e até 1 Mbit/s em
Classical CAN. CAN FD: IDs de 11 ou 29 bits, até 64 bytes; a fase de dados
pode usar bitrate maior com BRS. 2 ou 5 Mbit/s são configurações exemplificativas,
dependentes do controlador, transceptor, comprimento e projeto da rede.
"""
import csv
import math
import multiprocessing as mp
import os
import queue
import struct
import threading
import time
from pathlib import Path
from uuid import uuid4

import can
import numpy as np
import matplotlib.pyplot as plt
from apoio import salvar_csv, salvar_json, salvar_fig, ler_csv

SINAIS = {0x100: ('velocidade', 'km/h', '<HHI', .01),
          0x101: ('rpm', 'rpm', '<HHI', 1.),
          0x102: ('temperatura', '°C', '<hHI', .01)}
IDS = list(SINAIS)
FILTROS = [{'can_id': i, 'can_mask': 0x7FF, 'extended': False} for i in IDS]

def sinais_simulados(t):
    return {0x100: 45 + 12*math.sin(2*math.pi*t/5),
            0x101: 1600 + 550*math.sin(2*math.pi*t/5 + .15),
            0x102: 78 + 1.5*t + .4*math.sin(2*math.pi*t/3)}

def codificar(can_id, valor, seq, t_ms):
    _, _, fmt, escala = SINAIS[can_id]
    return struct.pack(fmt, round(valor/escala), seq % 65536, t_ms % 2**32)

def decodificar(msg):
    if msg.arbitration_id not in SINAIS or msg.is_extended_id or len(msg.data) != 8:
        raise ValueError('ID, formato ou tamanho de payload inesperado.')
    nome, unidade, fmt, escala = SINAIS[msg.arbitration_id]
    bruto, seq, t_ms = struct.unpack(fmt, msg.data)
    return nome, bruto*escala, unidade, seq, t_ms

def registro(msg, tx_mono):
    rx = time.monotonic_ns()
    nome, valor, unidade, seq, t_ms = decodificar(msg)
    return dict(timestamp_can_epoch_s=msg.timestamp, rx_epoch_ns=time.time_ns(),
                rx_mono_ns=rx, tx_mono_ns=tx_mono,
                atraso_software_ms=(rx-tx_mono)/1e6,
                id_hex=f'0x{msg.arbitration_id:03X}', sinal=nome, valor=valor,
                unidade=unidade, contador=seq, t_sensor_ms=t_ms,
                payload_hex=bytes(msg.data).hex(' '), pid_assinante=os.getpid())

def imprimir(row):
    print(f"{row['timestamp_can_epoch_s']:.6f} {row['id_hex']} "
          f"{row['sinal']}={row['valor']:.2f} {row['unidade']} "
          f"seq={row['contador']} atraso_SW={row['atraso_software_ms']:.3f} ms",
          flush=True)

def plotar_series(path_csv, out):
    rows = ler_csv(path_csv)
    fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, (can_id, (nome, unidade, _, _)) in zip(axs, SINAIS.items()):
        parte = [r for r in rows if r['id_hex'] == f'0x{can_id:03X}']
        ax.plot([float(r['t_sensor_ms'])/1000 for r in parte],
                [float(r['valor']) for r in parte], 'k.-', markersize=4)
        ax.set_ylabel(f'{nome}\n({unidade})'); ax.grid(alpha=.2)
    axs[-1].set_xlabel('Tempo do sensor simulado (s)')
    salvar_fig(fig, Path(out)/'series_can.png')

def resumo_can(out, hz):
    out = Path(out)
    rows = ler_csv(out/'mensagens_can.csv')
    partes = {}
    for i, (nome, unidade, _, _) in SINAIS.items():
        rs = [r for r in rows if r['id_hex'] == f'0x{i:03X}']
        rx = np.array([int(r['rx_mono_ns']) for r in rs], dtype=np.float64)
        dt = np.diff(rx)/1e6
        partes[nome] = dict(id=f'0x{i:03X}', mensagens=len(rs),
                            intervalo_rx_medio_ms=float(dt.mean()) if dt.size else None,
                            jitter_intervalo_rx_ms=float(dt.std()) if dt.size else None)
    atrasos = np.array([float(r['atraso_software_ms']) for r in rows])
    resumo = dict(total_mensagens=len(rows), hz_por_sinal=hz, periodo_alvo_ms=1000/hz,
                  sinais=partes, atraso_software_medio_ms=float(atrasos.mean()),
                  atraso_software_p95_ms=float(np.percentile(atrasos, 95)),
                  descricao='Tempos do software/IPC/VirtualBus; não são latência de CAN físico.')
    salvar_json(out/'resumo_can.json', resumo)
    plotar_series(out/'mensagens_can.csv', out)
    print('Resumo CAN:', resumo, flush=True)
    return resumo

def can_threads(out, ciclos=20, hz=10., log=True):
    """Nós virtuais no mesmo processo: publicação periódica e recepção em thread."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    channel = 'aula-'+uuid4().hex
    rows, publicados, tx_clock = [], [], {}
    erro = []
    with can.Bus(interface='virtual', channel=channel) as tx, \
         can.Bus(interface='virtual', channel=channel, can_filters=FILTROS) as rx:
        def publicar():
            try:
                t0 = time.monotonic_ns()
                for seq in range(ciclos):
                    alvo = t0+round(seq/hz*1e9)
                    time.sleep(max(0, (alvo-time.monotonic_ns())/1e9))
                    for i, valor in sinais_simulados(seq/hz).items():
                        data = codificar(i, valor, seq, round(seq/hz*1000))
                        tx_ns = time.monotonic_ns(); tx_clock[(i, seq)] = tx_ns
                        tx.send(can.Message(arbitration_id=i, data=data,
                                            is_extended_id=False, check=True))
                        publicados.append(dict(id_hex=f'0x{i:03X}', contador=seq,
                                               tx_mono_ns=tx_ns, pid_publicador=os.getpid()))
            except Exception as exc:
                erro.append(repr(exc))
        th = threading.Thread(target=publicar, daemon=True); th.start()
        limite = time.monotonic()+ciclos/hz+5
        while len(rows) < ciclos*3 and time.monotonic() < limite:
            msg = rx.recv(.1)
            if msg is not None:
                _, _, _, seq, _ = decodificar(msg)
                row = registro(msg, tx_clock[(msg.arbitration_id, seq)])
                rows.append(row)
                if log: imprimir(row)
        th.join(timeout=3)
    if erro or len(rows) != ciclos*3:
        raise RuntimeError(f'CAN incompleto: {len(rows)}/{ciclos*3}; {erro}')
    salvar_csv(out/'mensagens_can.csv', rows)
    salvar_csv(out/'publicados_can.csv', publicados)
    return resumo_can(out, hz)

def _publicador(q, pronto, ciclos, hz, out):
    """Processo publicador. tx→VirtualBus→gateway→ponte IPC."""
    channel = 'pub-'+uuid4().hex
    registros = []
    if not pronto.wait(15): raise TimeoutError('Assinante não ficou pronto.')
    with can.Bus(interface='virtual', channel=channel) as tx, \
         can.Bus(interface='virtual', channel=channel) as gateway:
        t0 = time.monotonic_ns()
        for seq in range(ciclos):
            alvo = t0+round(seq/hz*1e9)
            time.sleep(max(0, (alvo-time.monotonic_ns())/1e9))
            for i, valor in sinais_simulados(seq/hz).items():
                msg = can.Message(arbitration_id=i,
                                  data=codificar(i, valor, seq, round(seq/hz*1000)),
                                  is_extended_id=False, check=True)
                t_send = time.monotonic_ns()
                tx.send(msg)
                m = gateway.recv(1)
                if m is None: raise RuntimeError('Gateway publicador sem mensagem.')
                q.put((m.arbitration_id, bytes(m.data), m.timestamp, t_send))
                registros.append(dict(id_hex=f'0x{i:03X}', contador=seq,
                                       tx_mono_ns=t_send, pid_publicador=os.getpid()))
    q.put(None)
    salvar_csv(Path(out)/'publicados_can.csv', registros)

def _assinante(q, pronto, out):
    """Processo assinante. ponte IPC→gateway→VirtualBus filtrado→decodificação."""
    channel = 'sub-'+uuid4().hex
    out = Path(out); path = out/'mensagens_can.csv'
    with can.Bus(interface='virtual', channel=channel, preserve_timestamps=True) as gateway, \
         can.Bus(interface='virtual', channel=channel, can_filters=FILTROS) as rx, \
         path.open('w', newline='', encoding='utf-8') as f:
        writer = None
        pronto.set()
        while True:
            item = q.get(timeout=20)
            if item is None: break
            i, data, ts, tx_ns = item
            gateway.send(can.Message(arbitration_id=i, data=data, timestamp=ts,
                                     is_extended_id=False, check=True))
            msg = rx.recv(.2)
            if msg is None: continue  # Um ID fora dos filtros seria rejeitado aqui.
            row = registro(msg, tx_ns)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row); f.flush()
            imprimir(row)

def can_processos(out, ciclos=50, hz=10.):
    """Executa dois processos reais. spawn funciona no Windows/Linux e via notebook."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context('spawn'); q = ctx.Queue(maxsize=1024); ready = ctx.Event()
    sub = ctx.Process(target=_assinante, args=(q, ready, str(out)), name='assinante_CAN')
    pub = ctx.Process(target=_publicador, args=(q, ready, ciclos, hz, str(out)), name='publicador_CAN')
    try:
        sub.start(); pub.start()
        pub.join(timeout=ciclos/hz+20); sub.join(timeout=20)
        if pub.is_alive() or sub.is_alive() or pub.exitcode != 0 or sub.exitcode != 0:
            raise RuntimeError(f'Processos CAN: publicador={pub.exitcode}; assinante={sub.exitcode}')
        rows = ler_csv(out/'mensagens_can.csv')
        if len(rows) != ciclos*3:
            raise RuntimeError('Contagem final CAN não corresponde ao esperado.')
        pubs = ler_csv(out/'publicados_can.csv')
        salvar_json(out/'processos.json', dict(pid_publicador=int(pubs[0]['pid_publicador']),
                    pid_assinante=int(rows[0]['pid_assinante']),
                    ponte='multiprocessing.Queue entre gateways de VirtualBus locais',
                    interface='virtual', processos_distintos=True))
        return resumo_can(out, hz)
    finally:
        for p in (pub, sub):
            if p.is_alive(): p.terminate(); p.join(timeout=3)
        q.close(); q.join_thread()