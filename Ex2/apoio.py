"""Funções comuns dos exemplos de protocolos veiculares.

Instalação: python -m pip install -r requirements.txt.
No Colab, usar requirements-colab.txt. Os exemplos usam CPU e sockets locais.
Este módulo organiza arquivos, argumentos e gráficos; não exige rodar um
exercício anterior. Bibliotecas da biblioteca padrão não precisam de pip.
"""
from pathlib import Path
import argparse
import csv
import json
import platform
import sys
import time

import cv2
import can
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False})

def ambiente():
    cpu = platform.processor()
    p = Path('/proc/cpuinfo')
    if p.exists():
        cpu = next((l.split(':', 1)[1].strip() for l in p.read_text().splitlines()
                    if l.startswith('model name')), cpu)
    return {'cpu': cpu, 'sistema': platform.platform(), 'python': sys.version.split()[0],
            'python_can': can.__version__, 'opencv': cv2.__version__,
            'numpy': np.__version__, 'threads_opencv': cv2.getNumThreads(),
            'clock': time.get_clock_info('monotonic').implementation,
            'resolucao_clock_ns': time.get_clock_info('monotonic').resolution * 1e9,
            'local_da_medicao': 'Mesmo host; sockets IPv4 em loopback 127.0.0.1',
            'gpu': 'Não utilizada por estes exemplos'}

def iniciar(nome, frames=60, ciclos=20):
    p = argparse.ArgumentParser(description=Path(nome).stem)
    p.add_argument('--frames', type=int, default=frames)
    p.add_argument('--ciclos', type=int, default=ciclos)
    p.add_argument('--fps', type=float, default=30.)
    p.add_argument('--hz', type=float, default=10.)
    p.add_argument('--video', default=str(RAIZ/'dados/direcao_udacity.mp4'))
    p.add_argument('--camera', type=int, default=None)
    p.add_argument('--qualidade', type=int, default=70)
    p.add_argument('--datagrama', type=int, default=1200)
    p.add_argument('--mostrar', action='store_true')
    p.add_argument('--sem-gui', action='store_true', help='Salva figuras e feed; sem janelas.')
    p.add_argument('--log-cada', type=int, default=30)
    p.add_argument('--saida', default=None)
    a = p.parse_args()
    if a.frames < 1 or a.ciclos < 1 or a.fps <= 0 or a.hz <= 0:
        p.error('frames/ciclos/fps/hz devem ser positivos.')
    if not 1 <= a.qualidade <= 100 or not 256 <= a.datagrama <= 1400:
        p.error('Qualidade: 1–100; datagrama total: 256–1400 bytes.')
    cv2.setNumThreads(1)
    out = Path(a.saida) if a.saida else RAIZ/'resultados'/Path(nome).stem
    out.mkdir(parents=True, exist_ok=True)
    salvar_json(out/'ambiente.json', ambiente())
    return a, out

def salvar_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def salvar_csv(path, rows, fields=None):
    rows = list(rows)
    if not fields:
        fields = list(rows[0]) if rows else []
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def ler_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def salvar_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)

def quadro_inicial(a):
    source = a.camera if a.camera is not None else a.video
    cap = cv2.VideoCapture(source)
    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Não foi possível ler a fonte: {source}')
        return cv2.resize(frame, (640, 480))
    finally:
        cap.release()

def parametros_stream(a, **extras):
    d = dict(frames=a.frames, fps=a.fps, video=a.video, camera=a.camera,
             quality=a.qualidade, datagram_size=a.datagrama, show=a.mostrar,
             log_every=a.log_cada, width=640, height=480, clock='monotonic',
             drop_prob=0., seed=42, reorder=False, duplicates=False,
             tcp_nodelay=True, assembly_timeout_ms=200., deadline_ms=1000/30,
             sync_attested=False, write_video=True)
    d.update(extras)
    return d