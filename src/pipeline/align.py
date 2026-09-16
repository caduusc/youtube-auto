"""Onde cada beat visual caiu na fala real.

O storyboard nao escreve tempo — escreve um trecho do roteiro. Este modulo
descobre em que segmento do transcript aquele trecho foi dito, e e o que faz o
desenho sobreviver a voce improvisar na gravacao.

Duas coisas dominam o desenho.

A PRIMEIRA e que o sinal mais forte vem antes do mais esperto. Se voce seguiu
o roteiro, as palavras do ancora estao LITERALMENTE na fala transcrita — e uma
busca por sequencia de palavras acerta com certeza, de graca, e sem carregar o
torch. So quando a entrega parafraseou e que vale gastar embedding. Na pratica
isso quer dizer que o caminho comum nao paga nada.

A SEGUNDA e que o alinhamento e MONOTONICO. Os beats estao em ordem e o
transcript esta em ordem, entao o beat 7 nao pode casar antes do beat 6. Um
argmax por ancora, sem cursor, embaralharia as imagens — e o embaralhamento
seria plausivel demais para aparecer numa revisao rapida, porque cada imagem
isolada continuaria "parecendo certa".
"""

from __future__ import annotations

from .bank import cosine
from .config import AlignConfig
from .embed import Embedder
from .log import log
from .schemas import BeatPlacement, Script, Storyboard, Transcript
from .storyboard import collapse, normalize


def segment_words(transcript: Transcript) -> list[list[str]]:
    """As palavras normalizadas de cada segmento, para a busca literal."""
    return [collapse(normalize(s.text)).split() for s in transcript.segments]


def find_literal(anchor: list[str], words: list[list[str]], start: int) -> int:
    """Primeiro segmento, a partir de `start`, que contem a sequencia do ancora.

    A busca atravessa fronteira de segmento: o Whisper quebra por pausa, nao
    por frase, entao um ancora de seis palavras cai partido em dois segmentos
    com frequencia. Sem isso o caminho literal falharia justamente nos ancoras
    mais longos, que sao os mais confiaveis.
    """
    if not anchor:
        return -1

    for inicio in range(start, len(words)):
        # O ancora pode comecar em QUALQUER palavra deste segmento, entao
        # precisa haver `len(segmento) + len(ancora)` palavras acumuladas —
        # parar em `len(ancora)` era o bug: com o ancora comecando na ultima
        # palavra do segmento, faltava tudo o que vem do proximo.
        preciso = len(words[inicio]) + len(anchor)
        corrido: list[str] = []
        for i in range(inicio, len(words)):
            corrido += words[i]
            if len(corrido) >= preciso:
                break

        for offset in range(len(words[inicio]) or 1):
            if corrido[offset:offset + len(anchor)] == anchor:
                return inicio
    return -1


def place(
    storyboard: Storyboard,
    transcript: Transcript,
    *,
    rules: AlignConfig,
    embedder: Embedder | None = None,
) -> list[BeatPlacement]:
    """Onde cada beat entra, em indice de segmento. Monotonico."""
    words = segment_words(transcript)
    n = len(words)
    cursor = 0
    colocados: list[BeatPlacement] = []

    # Os embeddings dos segmentos sao calculados sob demanda: se todo ancora
    # casar literalmente, o embedder nunca e tocado — e ele arrasta o torch.
    vetores: list[list[float] | None] = [None] * n

    def vetor(i: int) -> list[float]:
        if vetores[i] is None:
            vetores[i] = embedder.embed(transcript.segments[i].text)
        return vetores[i]

    for beat in storyboard.beats:
        alvo = collapse(normalize(beat.script_anchor)).split()

        indice = find_literal(alvo, words, cursor)
        if indice >= 0:
            colocados.append(BeatPlacement(
                beat_id=beat.beat_id, anchor_offset=beat.anchor_offset,
                segment=indice, method="literal", similarity=1.0,
            ))
            cursor = indice + 1
            continue

        if embedder is None:
            colocados.append(BeatPlacement(
                beat_id=beat.beat_id, anchor_offset=beat.anchor_offset,
                segment=-1, method="unaligned", similarity=0.0,
            ))
            continue

        alvo_vetor = embedder.embed(beat.script_anchor)
        melhor, melhor_score = -1, 0.0
        for i in range(cursor, n):
            score = cosine(alvo_vetor, vetor(i))
            if score > melhor_score:
                melhor, melhor_score = i, score

        if melhor < 0 or melhor_score < rules.min_similarity:
            colocados.append(BeatPlacement(
                beat_id=beat.beat_id, anchor_offset=beat.anchor_offset,
                segment=-1, method="unaligned", similarity=round(melhor_score, 4),
            ))
            continue

        colocados.append(BeatPlacement(
            beat_id=beat.beat_id, anchor_offset=beat.anchor_offset,
            segment=melhor, method="semantic", similarity=round(melhor_score, 4),
        ))
        cursor = melhor + 1

    literais = sum(1 for c in colocados if c.method == "literal")
    semanticos = sum(1 for c in colocados if c.method == "semantic")
    orfaos = sum(1 for c in colocados if c.method == "unaligned")
    log("align.ok", beats=len(colocados), literais=literais,
        semanticos=semanticos, orfaos=orfaos,
        embedder="nao carregado" if semanticos == 0 and orfaos == 0 else "usado")
    return colocados


def spans_for(
    placements: list[BeatPlacement],
    storyboard: Storyboard,
    transcript: Transcript,
) -> list[tuple[int, int, int]]:
    """(beat_id, primeiro_segmento, ultimo_segmento) de cada faixa de b-roll.

    A faixa comeca no segmento do ancora e vai crescendo em SEGMENTO INTEIRO
    ate cobrir o tempo de tela que o beat pede. Crescer por segmento e nao por
    tempo e o que preserva "nunca cortar no meio da frase": a fronteira e
    sempre fronteira de segmento, entao cortar no meio e inexprimivel — nao
    validado depois, inexprimivel.
    """
    por_id = {b.beat_id: b for b in storyboard.beats}
    n = len(transcript.segments)
    faixas: list[tuple[int, int, int]] = []
    ocupado_ate = -1

    for colocado in placements:
        if colocado.segment < 0:
            continue
        beat = por_id[colocado.beat_id]
        inicio = max(colocado.segment, ocupado_ate + 1)
        if inicio >= n:
            continue

        fim = inicio
        coberto = transcript.segments[inicio].duration
        while coberto < beat.screen_seconds and fim + 1 < n:
            fim += 1
            coberto += transcript.segments[fim].duration

        faixas.append((colocado.beat_id, inicio, fim))
        ocupado_ate = fim

    return faixas
