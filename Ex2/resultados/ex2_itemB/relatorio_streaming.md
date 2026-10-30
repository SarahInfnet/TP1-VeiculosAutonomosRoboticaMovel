# Relatório do streaming TCP/UDP

## Ambiente e método
Dois processos reais (produtor/consumidor), sockets IPv4 em 127.0.0.1, mesma
máquina. Vídeo real de direção repetido ao chegar ao fim, 640×480, JPEG
qualidade 70, 30.0 FPS-alvo,
320 frames por protocolo. Clock monotônico comum aos processos.
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
| UDP | 320 | 320 | 2.2098 | 5.5809 | 3.4806 | 0.00 |
| TCP | 320 | 320 | 3.9529 | 4.2739 | 7.7607 | 0.00 |

## Interpretação e decisão
Neste ensaio, UDP apresentou a menor latência média, TCP o menor
jitter e UDP o menor p99. UDP teve 3 frames
recebidos acima do prazo de idade de 33.33 ms; TCP teve
12. Essa contagem não inclui frames perdidos.
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
