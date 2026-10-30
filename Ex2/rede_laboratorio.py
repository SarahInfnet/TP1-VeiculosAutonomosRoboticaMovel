"""Streaming didático com sockets reais, JPEG e processos separados.

Instalar OpenCV, NumPy e Matplotlib via requirements.txt; socket, struct,
multiprocessing, time e zlib pertencem à biblioteca padrão do Python.
No Colab, utilizar requirements-colab.txt e o feed MP4 em vez de cv2.imshow.

O ensaio padrão usa 127.0.0.1, uma máquina e time.monotonic_ns() comum aos
processos. A latência principal vai do instante anterior ao primeiro envio
até o fim da decodificação JPEG. Não é atraso exclusivo do meio físico.
Também são registrados transporte/reassembly, decode e idade desde captura.
Jitter neste laboratório = desvio padrão populacional das latências (ddof=0).
TCP é um fluxo: recv(n) pode devolver menos de n bytes; usar recv_exact.
UDP fragmenta o JPEG na aplicação em datagramas de até 1200 bytes totais,
remonta por frame/índice e expira mensagens incompletas. Não retransmite.
Perdas artificiais têm contadores separados das perdas observadas no socket.
"""
import argparse
import math
import multiprocessing as mp
import random
import socket
import struct
import time
import zlib
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import matplotlib.pyplot as plt
from apoio import ambiente, salvar_csv, salvar_json, salvar_fig, ler_csv, RAIZ

TCP_H = struct.Struct('!4sIIQQI')  # magic, seq, bytes, envio_ns, captura_ns, crc32
UDP_H = struct.Struct('!4sIHHQQII') # magic, seq, índice, total, envio, captura, bytes, crc
MAX_JPEG = 4*1024*1024

def relogio(conf):
    return time.perf_counter_ns() if conf['clock']=='monotonic' else time.time_ns()

def recv_exact(sock, n):
    """Retorna exatamente n bytes; EOF antes da quantidade completa é erro."""
    partes = bytearray()
    while len(partes) < n:
        trecho = sock.recv(n-len(partes))
        if not trecho: raise EOFError(f'Conexão encerrada após {len(partes)}/{n} bytes.')
        partes.extend(trecho)
    return bytes(partes)

def fragmentar(jpeg, seq, ts_envio, ts_captura, datagrama=1200):
    if not 256 <= datagrama <= 1400 or not 0 < len(jpeg) <= MAX_JPEG:
        raise ValueError('Tamanho de datagrama ou JPEG inválido.')
    capacidade = datagrama-UDP_H.size
    n = math.ceil(len(jpeg)/capacidade)
    if n > 65535: raise ValueError('Número de fragmentos excede o cabeçalho.')
    crc = zlib.crc32(jpeg)
    return [UDP_H.pack(b'JPG1', seq, i, n, ts_envio, ts_captura, len(jpeg), crc)
            +jpeg[i*capacidade:(i+1)*capacidade] for i in range(n)]

def ler_datagrama(packet):
    if len(packet) <= UDP_H.size: raise ValueError('Datagrama curto.')
    magic, seq, idx, n, tx, captura, tamanho, crc = UDP_H.unpack(packet[:UDP_H.size])
    if magic != b'JPG1' or n < 1 or idx >= n or not 0 < tamanho <= MAX_JPEG:
        raise ValueError('Cabeçalho UDP inválido.')
    return (seq, idx, n, tx, captura, tamanho, crc), packet[UDP_H.size:]

class Feed:
    """Grava apenas os frames efetivamente recebidos/decodificados pelo cliente."""
    def __init__(self, out, fps, enabled=True):
        self.out = Path(out); self.fps = fps; self.enabled = enabled
        self.writer = None; self.n = 0
        self.temp = self.out/('_gravacao_'+uuid4().hex+'.mp4')
    def escrever(self, frame, seq, latencia):
        if not self.enabled: return
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.temp), cv2.VideoWriter_fourcc(*'mp4v'),
                                         self.fps, (w, h))
            if not self.writer.isOpened(): raise RuntimeError('VideoWriter não abriu.')
        im = frame.copy()
        label = f'Frame recebido: {seq}'
        if latencia is not None: label += f' | latencia: {latencia:.2f} ms'
        cv2.rectangle(im, (0, 0), (im.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(im, label, (9, 25), cv2.FONT_HERSHEY_SIMPLEX, .53, (255,255,255), 1)
        self.writer.write(im); self.n += 1
        if self.n == 1: cv2.imwrite(str(self.out/'primeiro_frame_recebido.png'), im)
        self.last = im
    def fechar(self):
        if self.writer is None: return
        self.writer.release()
        cv2.imwrite(str(self.out/'ultimo_frame_recebido.png'), self.last)
        cap = cv2.VideoCapture(str(self.temp))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, _ = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0,total-1)); ultimo, _ = cap.read()
        finally: cap.release()
        if total != self.n or not ok or not ultimo:
            raise RuntimeError(f'Feed inválido: {total} gravados/{self.n} recebidos.')
        self.temp.replace(self.out/'feed_recebido.mp4')


def _receber_jpeg(jpeg, seq, tx, captura, rx_final, conf, rows, feed):
    t0_decode = time.monotonic_ns()

    frame = cv2.imdecode(
        np.frombuffer(jpeg, np.uint8),
        cv2.IMREAD_COLOR
    )

    decode_end = relogio(conf)
    dt_decode = (time.monotonic_ns() - t0_decode) / 1e6

    if frame is None:
        return False

    clock_valido = (
        conf['clock'] == 'monotonic'
        or conf.get('sync_attested', False)
    )

    latencia = (decode_end - tx) / 1e6 if clock_valido else None
    transporte = (rx_final - tx) / 1e6 if clock_valido else None
    idade = (decode_end - captura) / 1e6 if clock_valido else None

    # Diagnostico: verificar os timestamps do primeiro frame
    if seq == 0:
        print("\n=== DEBUG LATENCIA ===")
        print("Timestamp envio:", tx)
        print("Timestamp recebimento:", decode_end)
        print("Diferenca (ns):", decode_end - tx)
        print("Latencia (ms):", latencia)
        print("======================\n", flush=True)

    if latencia is not None and latencia < 0:
        raise RuntimeError(
            'Latência negativa: conferir domínio/sincronização dos relógios.'
        )

    rows.append(dict(
        frame_id=seq,
        envio_ns=tx,
        captura_ns=captura,
        recepcao_completa_ns=rx_final,
        decode_fim_ns=decode_end,
        latencia_ms=latencia,
        transporte_reassembly_ms=transporte,
        decode_ms=dt_decode,
        idade_frame_ms=idade,
        jpeg_bytes=len(jpeg),
        acima_prazo=int(idade > conf['deadline_ms']) if idade is not None else '',
        pid_cliente=mp.current_process().pid
    ))

    if seq % max(1, conf['log_every']) == 0:
        l = f'{latencia:.3f} ms' if latencia is not None else 'indisponível (relógios)'
        print(
            f"Cliente {conf['protocol'].upper()}: "
            f"frame={seq} latência={l}",
            flush=True
        )

    feed.escrever(frame, seq, latencia)

    if conf.get('show'):
        cv2.imshow('Cliente: camera recebida', frame)
        cv2.waitKey(1)

    return True


def _enviar_frames(conf, sock, destino=None):
    source = conf['camera'] if conf.get('camera') is not None else conf['video']
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): raise RuntimeError(f'Fonte não abriu: {source}')
    rows = []; rng = random.Random(conf['seed']); t0 = time.monotonic_ns()
    try:
        for seq in range(conf['frames']):
            alvo = t0+round(seq/conf['fps']*1e9)
            time.sleep(max(0, (alvo-time.monotonic_ns())/1e9))
            captura = relogio(conf)
            begin = time.monotonic_ns(); ok, frame = cap.read()
            if not ok and conf.get('camera') is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, frame = cap.read()
            if not ok: raise RuntimeError('Fim/falha de captura antes de todos os frames.')
            read_ms = (time.monotonic_ns()-begin)/1e6
            begin = time.monotonic_ns()
            frame = cv2.resize(frame, (conf['width'], conf['height']))
            resize_ms = (time.monotonic_ns()-begin)/1e6
            begin = time.monotonic_ns()
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY,conf['quality']])
            if not ok: raise RuntimeError('Falha na codificação JPEG.')
            jpeg = buf.tobytes(); encode_ms = (time.monotonic_ns()-begin)/1e6
            crc = zlib.crc32(jpeg)
            tx = relogio(conf); api_begin = time.monotonic_ns()
            planned, sent, dropped, duplicated = 0,0,0,0
            if conf['protocol']=='tcp':
                header = TCP_H.pack(b'JPG1', seq, len(jpeg), tx, captura, crc)
                sock.sendall(header+jpeg)
            else:
                packets = fragmentar(jpeg, seq, tx, captura, conf['datagram_size'])
                planned = len(packets)
                indices = list(range(planned))
                if conf.get('reorder'): rng.shuffle(indices)
                kept = []
                for idx in indices:
                    if rng.random() < conf['drop_prob']:
                        dropped += 1; continue # Perda artificial antes do sendto.
                    sock.sendto(packets[idx], destino); sent += 1; kept.append(idx)
                if conf.get('duplicates') and kept:
                    sock.sendto(packets[kept[0]], destino); duplicated = 1
            rows.append(dict(frame_id=seq, captura_ns=captura, envio_ns=tx,
                             jpeg_bytes=len(jpeg), read_ms=read_ms, resize_ms=resize_ms,
                             encode_ms=encode_ms, envio_api_ms=(time.monotonic_ns()-api_begin)/1e6,
                             fragmentos_planejados=planned, datagramas_unicos_enviados=sent,
                             datagramas_descartados_artificialmente=dropped,
                             datagramas_duplicados_enviados=duplicated,
                             atraso_agenda_ms=max(0,(api_begin-alvo)/1e6),
                             pid_servidor=mp.current_process().pid))
        if conf['protocol']=='tcp':
            sock.sendall(TCP_H.pack(b'FIM1',len(rows),0,0,0,0))
        else:
            for _ in range(3):
                sock.sendto(b'FIM1'+struct.pack('!I',len(rows)),destino)
                time.sleep(.01)
    finally:
        cap.release()
        salvar_csv(Path(conf['out'])/'transmitidos.csv', rows)
    print(f"Servidor {conf['protocol'].upper()}: {len(rows)} frames transmitidos.", flush=True)

def produtor(conf, ready=None):
    cv2.setNumThreads(1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM if conf['protocol']=='tcp'
                       else socket.SOCK_DGRAM) as sock:
        sock.settimeout(15)
        if conf['protocol']=='tcp':
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            sock.bind((conf.get('host','127.0.0.1'),conf.get('port',0)))
            sock.listen(1)
            if ready is not None: ready.put(sock.getsockname()[1])
            print('Servidor TCP pronto:',sock.getsockname(),flush=True)
            conn, _ = sock.accept()
            with conn:
                conn.settimeout(15)
                conn.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,int(conf['tcp_nodelay']))
                _enviar_frames(conf,conn)
        else:
            _enviar_frames(conf,sock,(conf.get('host','127.0.0.1'),conf['port']))

def consumidor_tcp(conf):
    cv2.setNumThreads(1); rows=[]; seqs=set(); out=Path(conf['out'])
    feed=Feed(out,conf['fps'],conf['write_video']); expected=None
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
        sock.settimeout(15)
        sock.connect((conf.get('host','127.0.0.1'),conf['port']))
        sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,int(conf['tcp_nodelay']))
        try:
            while True:
                magic,seq,n,tx,captura,crc = TCP_H.unpack(recv_exact(sock,TCP_H.size))
                if magic==b'FIM1' and n==0: expected=seq; break
                if magic!=b'JPG1' or not 0 < n <= MAX_JPEG: raise ValueError('Header TCP inválido.')
                jpeg=recv_exact(sock,n); rx_final=relogio(conf)
                if zlib.crc32(jpeg)!=crc: raise ValueError('CRC do JPEG TCP divergente.')
                if seq not in seqs:
                    if _receber_jpeg(jpeg,seq,tx,captura,rx_final,conf,rows,feed): seqs.add(seq)
        finally:
            feed.fechar(); salvar_csv(out/'recebidos.csv',rows)
            if conf.get('show'): cv2.destroyAllWindows()
    salvar_json(out/'cliente_contadores.json',dict(frames_fim_servidor=expected,
                frames_decodificados=len(rows),datagramas_unicos_recebidos=None))

def consumidor_udp(conf, ready=None):
    cv2.setNumThreads(1); out=Path(conf['out']); rows=[]; pacotes=[]
    pending={}; completos=set(); expirados=set(); seen=set(); invalidos=0
    feed=Feed(out,conf['fps'],conf['write_video']); fim=None; expected=None
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,4*1024*1024)
        sock.bind((conf.get('host','127.0.0.1'),conf.get('port',0))); sock.settimeout(.03)
        if ready is not None: ready.put(sock.getsockname()[1])
        print('Cliente UDP pronto:',sock.getsockname(),flush=True)
        inicio=time.monotonic(); global_limit=conf['frames']/conf['fps']*2+15
        try:
            while time.monotonic()-inicio < global_limit:
                agora=time.monotonic_ns()
                for seq,entry in list(pending.items()):
                    if (agora-entry['inicio'])/1e6 > conf['assembly_timeout_ms']:
                        expirados.add(seq); del pending[seq]
                if fim is not None and time.monotonic() > fim+.35: break
                try: packet,_=sock.recvfrom(65535)
                except socket.timeout: continue
                rx_final=relogio(conf)
                if len(packet)==8 and packet[:4]==b'FIM1':
                    expected=struct.unpack('!I',packet[4:])[0]
                    if fim is None: fim=time.monotonic()
                    continue
                try: meta,body=ler_datagrama(packet)
                except ValueError: invalidos+=1; continue
                seq,idx,n,tx,captura,tamanho,crc=meta
                duplicate=(seq,idx) in seen
                pacotes.append(dict(frame_id=seq,fragmento=idx,total=n,
                                    recepcao_ns=rx_final,bytes=len(packet),duplicado=int(duplicate)))
                seen.add((seq,idx))
                if seq in completos or seq in expirados or duplicate: continue
                if seq not in pending:
                    if len(pending)>=128:
                        oldest=min(pending,key=lambda s:pending[s]['inicio'])
                        expirados.add(oldest); del pending[oldest]
                    pending[seq]=dict(meta=meta,inicio=time.monotonic_ns(),partes={})
                entry=pending[seq]
                if (entry['meta'][2:] != meta[2:]): invalidos+=1; continue
                entry['partes'][idx]=body
                if len(entry['partes'])==n:
                    jpeg=b''.join(entry['partes'][i] for i in range(n)); del pending[seq]
                    if len(jpeg)!=tamanho or zlib.crc32(jpeg)!=crc:
                        invalidos+=1; expirados.add(seq); continue
                    if _receber_jpeg(jpeg,seq,tx,captura,rx_final,conf,rows,feed): completos.add(seq)
        finally:
            expirados.update(pending)
            feed.fechar(); salvar_csv(out/'recebidos.csv',rows)
            salvar_csv(out/'pacotes_udp.csv',pacotes)
            if conf.get('show'): cv2.destroyAllWindows()
        salvar_json(out/'cliente_contadores.json',dict(frames_fim_servidor=expected,
                    frames_decodificados=len(rows),datagramas_unicos_recebidos=len(seen),
                    datagramas_duplicados_recebidos=len(pacotes)-len(seen),
                    frames_incompletos_expirados=len(expirados),invalidos=invalidos,
                    rcvbuf_bytes=sock.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF)))
        if expected is None: raise RuntimeError('Mensagem final UDP não recebida; ensaio incompleto.')

def estatisticas(out,conf):
    out=Path(out); tx=ler_csv(out/'transmitidos.csv'); rx=ler_csv(out/'recebidos.csv')
    enviados={int(r['frame_id']) for r in tx}; recebidos={int(r['frame_id']) for r in rx}
    perdidos=sorted(enviados-recebidos)
    if len(enviados)!=conf['frames']: raise RuntimeError('Total transmitido diferente do solicitado.')
    validos=np.array([float(r['latencia_ms']) for r in rx if r.get('latencia_ms') not in ('',None)])
    idade=np.array([float(r['idade_frame_ms']) for r in rx if r.get('idade_frame_ms') not in ('',None)])
    summary=dict(protocolo=conf['protocol'].upper(),transmitidos=len(enviados),
                 recebidos=len(recebidos),frames_perdidos=len(perdidos),
                 perda_frames_pct=100*len(perdidos)/len(enviados),
                 media_latencia_ms=float(validos.mean()) if validos.size else None,
                 jitter_std_ms=float(validos.std(ddof=0)) if validos.size else None,
                 p95_latencia_ms=float(np.percentile(validos,95)) if validos.size else None,
                 p99_latencia_ms=float(np.percentile(validos,99)) if validos.size else None,
                 media_idade_frame_ms=float(idade.mean()) if idade.size else None,
                 acima_prazo_frames=int((idade>conf['deadline_ms']).sum()),
                 prazo_ms=conf['deadline_ms'],jpeg_medio_bytes=float(np.mean([int(r['jpeg_bytes']) for r in tx])),
                 taxa_payload_mbit_s=float(np.mean([int(r['jpeg_bytes']) for r in tx]))*8*conf['fps']/1e6,
                 escopo_latencia='Antes do primeiro envio → fim de imdecode; captura/encode excluídos',
                 jitter_definicao='Desvio padrão populacional das latências dos frames recebidos (ddof=0)',
                 perda_definicao='IDs transmitidos que não foram decodificados pelo cliente',
                 clock=conf['clock'],clock_entre_hosts='Exige sincronização e incerteza documentadas',
                 frames_perdidos_ids=perdidos,ambiente=ambiente(),config=conf)
    counters=__import__('json').loads((out/'cliente_contadores.json').read_text())
    if conf['protocol']=='udp':
        planned=sum(int(r['fragmentos_planejados']) for r in tx)
        sent=sum(int(r['datagramas_unicos_enviados']) for r in tx)
        artificial=sum(int(r['datagramas_descartados_artificialmente']) for r in tx)
        received=counters['datagramas_unicos_recebidos']
        summary.update(datagramas_planejados=planned,datagramas_unicos_enviados=sent,
                       datagramas_unicos_recebidos=received,drops_artificiais=artificial,
                       perda_datagramas_observada_pct=100*max(0,sent-received)/sent if sent else None,
                       perda_datagramas_total_pct=100*(planned-received)/planned if planned else None)
    else:
        summary['perda_pacotes_tcp']='Não inferida pelo aplicativo; retransmissões não medidas.'
    salvar_json(out/'resumo.json',summary)
    salvar_csv(out/'frames_perdidos.csv',[dict(frame_id=i) for i in perdidos],['frame_id'])
    if validos.size:
        fig,ax=plt.subplots(figsize=(8,4));ax.hist(validos,bins=25,color='0.65',edgecolor='black')
        ax.set(xlabel='Latência envio → decode (ms)',ylabel='Frames recebidos',title=conf['protocol'].upper())
        salvar_fig(fig,out/'histograma_latencia.png')
    print('Resumo streaming:',{k:summary[k] for k in ['protocolo','transmitidos','recebidos',
          'media_latencia_ms','jitter_std_ms','perda_frames_pct']},flush=True)
    return summary

def rodar_stream(protocol,out,config):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    conf=dict(config,protocol=protocol,out=str(out),host='127.0.0.1',port=0)
    ctx=mp.get_context('spawn');ready=ctx.Queue();processes=[]
    try:
        if protocol=='tcp':
            server=ctx.Process(target=produtor,args=(conf,ready),name='servidor_TCP')
            processes.append(server);server.start();port=ready.get(timeout=15)
            conf['port']=port
            client=ctx.Process(target=consumidor_tcp,args=(conf,),name='cliente_TCP')
            processes.append(client);client.start()
        elif protocol=='udp':
            client=ctx.Process(target=consumidor_udp,args=(conf,ready),name='cliente_UDP')
            processes.append(client);client.start();port=ready.get(timeout=15)
            conf['port']=port
            server=ctx.Process(target=produtor,args=(conf,),name='servidor_UDP')
            processes.append(server);server.start()
        else: raise ValueError('Escolha tcp ou udp.')
        limite=config['frames']/config['fps']*2+30
        server.join(timeout=limite);client.join(timeout=limite)
        if any(p.is_alive() or p.exitcode!=0 for p in processes):
            raise RuntimeError(f'Processos streaming: {[p.exitcode for p in processes]}')
        return estatisticas(out,conf)
    finally:
        for p in processes:
            if p.is_alive(): p.terminate();p.join(timeout=3)
        ready.close();ready.join_thread()

def comparar(out,config,ordem=('udp','tcp')):
    out=Path(out);result={}
    for protocol in ordem:
        result[protocol]=rodar_stream(protocol,out/protocol,config)
    tabela=[{k:r[k] for k in ['protocolo','transmitidos','recebidos','media_latencia_ms',
                              'jitter_std_ms','p95_latencia_ms','frames_perdidos','perda_frames_pct']}
            for r in [result['udp'],result['tcp']]]
    salvar_csv(out/'tabela_tcp_udp.csv',tabela);salvar_json(out/'comparacao.json',result)
    dados={p:np.array([float(r['latencia_ms']) for r in ler_csv(out/p/'recebidos.csv')
                      if r['latencia_ms']!='']) for p in ('udp','tcp')}
    all_values=np.concatenate(list(dados.values()));bins=np.histogram_bin_edges(all_values,bins=30)
    fig,ax=plt.subplots(figsize=(9,4.7))
    for p,ls in [('udp','-'),('tcp','--')]:
        ax.hist(dados[p],bins=bins,histtype='step',density=False,label=p.upper(),
                color='black',linestyle=ls,linewidth=1.8)
    ax.set(xlabel='Latência envio → fim da decodificação JPEG (ms)',ylabel='Frames recebidos')
    ax.legend();ax.grid(alpha=.16);salvar_fig(fig,out/'histograma_udp_tcp.png')
    conclusao(out,result)
    return result

def conclusao(out,result):
    udp,tcp=result['udp'],result['tcp']
    menor=min((udp,tcp),key=lambda r:r['media_latencia_ms'])['protocolo']
    menor_jitter=min((udp,tcp),key=lambda r:r['jitter_std_ms'])['protocolo']
    menor_p99=min((udp,tcp),key=lambda r:r['p99_latencia_ms'])['protocolo']
    text=f'''# Relatório do streaming TCP/UDP

## Ambiente e método
Dois processos reais (produtor/consumidor), sockets IPv4 em 127.0.0.1, mesma
máquina. Vídeo real de direção repetido ao chegar ao fim, 640×480, JPEG
qualidade {udp['config']['quality']}, {udp['config']['fps']} FPS-alvo,
{udp['transmitidos']} frames por protocolo. Clock monotônico comum aos processos.
Sem GUI; gravação do feed depois do timestamp de decode, podendo afetar filas
dos frames seguintes. Não há transceptor CAN nem Ethernet automotiva neste teste.

## Definições
Latência principal: t_fim_decode − t_antes_envio. Captura e JPEG excluídos;
empacotamento, transferência, escalonamento, remontagem e decode incluídos. Idade do frame:
t_fim_decode − t_inicio_captura. Jitter = std(latências), ddof=0. Perda de frames:
(N_transmitidos − N_decodificados_únicos)/N_transmitidos × 100. Percentis apenas
dos recebidos; perda reportada separadamente. Não são perdas de segmentos TCP.

## Resultados medidos
| Protocolo | Tx | Rx | Média (ms) | Jitter (ms) | p95 (ms) | Perda de frames (%) |
|---|---:|---:|---:|---:|---:|---:|
| UDP | {udp['transmitidos']} | {udp['recebidos']} | {udp['media_latencia_ms']:.4f} | {udp['jitter_std_ms']:.4f} | {udp['p95_latencia_ms']:.4f} | {udp['perda_frames_pct']:.2f} |
| TCP | {tcp['transmitidos']} | {tcp['recebidos']} | {tcp['media_latencia_ms']:.4f} | {tcp['jitter_std_ms']:.4f} | {tcp['p95_latencia_ms']:.4f} | {tcp['perda_frames_pct']:.2f} |

## Interpretação e decisão
Neste ensaio, {menor} apresentou a menor latência média, {menor_jitter} o menor
jitter e {menor_p99} o menor p99. UDP teve {udp['acima_prazo_frames']} frames
recebidos acima do prazo de idade de {udp['prazo_ms']:.2f} ms; TCP teve
{tcp['acima_prazo_frames']}. Essa contagem não inclui frames perdidos.
Os resultados descrevem
este host e esta execução; não demonstram que um protocolo sempre será mais
rápido em rede física. UDP permite descartar frames incompletos sem esperar a
recuperação de bytes anteriores; TCP entrega um fluxo confiável e ordenado,
podendo reter dados novos durante a recuperação de dados anteriores.

Para o foco deste exercício, câmera usada por percepção com prioridade à
atualidade e tolerância definida a lacunas, a escolha inicial é UDP com IDs,
remontagem limitada, política para frames ausentes, supervisão e
controle de carga. Deve ser validada em Ethernet veicular e sob interferência
de tráfego. Para gravação/transferência que exige todos os bytes, TCP é uma
escolha adequada. Confiabilidade do fluxo não garante cumprimento do prazo.
Os dados basais mostram a capacidade observada de cada alternativa e seus
trade-offs; a escolha não se apoia na suposição de que UDP sempre tem menor
média. Se o consumidor não tolerar lacunas, rever a escolha e a recuperação.

Uma rede de produção requer orçamento de atraso, sincronização, capacidade,
prioridades e, quando pertinente, TSN. Nenhum socket Python garante tempo real
determinístico. Repetir com ordens alternadas, rede física, carga concorrente,
duração maior, p95/p99 e medição de offset dos relógios. Não calcular latência
unidirecional entre PCs sem relógios sincronizados e incerteza conhecida.

## Evidências
transmitidos.csv, recebidos.csv, pacotes_udp.csv, resumo.json, comparação,
histograma e feeds recebidos. Perdas artificiais, quando usadas, aparecem
em drops_artificiais e são distintas da perda observada dos datagramas enviados.
'''
    (Path(out)/'relatorio_streaming.md').write_text(text,encoding='utf-8')

def servico_cli(role):
    p=argparse.ArgumentParser(description=role+' de câmera JPEG TCP/UDP')
    p.add_argument('--protocolo',choices=['udp','tcp'],default='udp')
    p.add_argument('--host',default='127.0.0.1',help='TCP servidor/UDP cliente: bind; outro papel: destino.')
    p.add_argument('--porta',type=int,default=5005)
    p.add_argument('--frames',type=int,default=320);p.add_argument('--fps',type=float,default=30)
    p.add_argument('--video',default=str(RAIZ/'dados/direcao_udacity.mp4'))
    p.add_argument('--camera',type=int,default=None);p.add_argument('--qualidade',type=int,default=70)
    p.add_argument('--datagrama',type=int,default=1200);p.add_argument('--mostrar',action='store_true')
    p.add_argument('--saida',default=str(RAIZ/'resultados'/'dois_terminais'))
    p.add_argument('--relogio',choices=['monotonic','epoch'],default='monotonic')
    p.add_argument('--relogios-sincronizados',action='store_true')
    args=p.parse_args();out=Path(args.saida);out.mkdir(parents=True,exist_ok=True)
    if args.frames<1 or args.fps<=0 or not 1<=args.qualidade<=100:
        p.error('Confira frames, fps e qualidade.')
    if not 256<=args.datagrama<=1400: p.error('Datagrama deve ter 256–1400 bytes.')
    conf=dict(protocol=args.protocolo,host=args.host,port=args.porta,frames=args.frames,
              fps=args.fps,video=args.video,camera=args.camera,quality=args.qualidade,
              datagram_size=args.datagrama,out=str(out),show=args.mostrar,log_every=30,
              clock=args.relogio,sync_attested=args.relogios_sincronizados,seed=42,
              drop_prob=0.,reorder=False,duplicates=False,tcp_nodelay=True,width=640,
              height=480,assembly_timeout_ms=200.,deadline_ms=1000/30,write_video=True)
    salvar_json(out/(role+'_config.json'),conf)
    if role=='servidor': produtor(conf)
    elif conf['protocol']=='udp': consumidor_udp(conf)
    else: consumidor_tcp(conf)
    print('Arquivos gravados em:',out,flush=True)