"""Deteccao de trecho morto: pausa longa e hesitacao.

A regra que mais importa aqui e negativa. Em portugues, varios sons de
hesitacao sao tambem palavras de conteudo:

    "isso E muito importante"     -> verbo, cortar destroi a frase
    "entao... E... eu acho que"   -> hesitacao, cortar melhora
    "UM problema serio"           -> artigo
    "bom, UM..., vamos ver"       -> hesitacao

Por isso a lista de `fillers` do config nunca decide sozinha. Todo candidato
passa por dois testes: silencio ANTES, e duracao acima do que aquele mesmo
token costuma durar nesta fala.

O silencio e medido so ANTES, e a assimetria e a parte que nao e obvia.
Hesitacao e pausa e som tem uma ordem fonetica fixa: o falante para, o som
preenche a parada, e a fala retoma sem intervalo.

    "entao ... E eu acho que"        silencio antes, nada depois  -> hesitacao
    "o problema E ... que ninguem"   nada antes, silencio depois  -> verbo

Os dois padroes sao o espelho um do outro, entao exigir silencio nos dois
lados rejeita os dois. Medido na saida real do faster-whisper: o intervalo
DEPOIS e exatamente 0.00s em 19 de 20 candidatos, porque o modelo atribui
spans de palavra contiguos em vez de silencio medido. Exigir aquele lado
nao era conservador, era uma regra que nunca podia disparar.

O que sobra para distinguir hesitacao de palavra iniciando frase depois de
uma pausa ("...olha isso. E importante que...") e a duracao, comparada com a
mediana daquele token no resto da fala: hesitacao e alongada, verbo nao.
A calibracao sai da propria fala, entao acompanha o ritmo de quem gravou.
Isso cai exatamente onde precisa: os tokens ambiguos ("e", "a", "um") sao os
frequentes, e tem amostra sobrando; os inequivocos ("hm", "ahn") sao raros e
caem no piso absoluto de `filler_min_seconds`, que para eles basta.
"""

from __future__ import annotations

import unicodedata

from .config import Config, TrimConfig
from .schemas import Cut, KeepRange, Transcript, TranscriptSegment, TrimPlan, TrimStats, Word
from .util import text_hash


def normalize(token: str) -> str:
    """Minusculas, sem acento e sem pontuacao, para casar com a lista.

    O acento cai de proposito: o Whisper alterna entre "e", "e'" e "eh" para
    o mesmo som, e manter acento exigiria listar todas as variantes.
    """
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFKD", token.lower())
        if not unicodedata.combining(ch)
    )
    return "".join(ch for ch in stripped if ch.isalnum())


def all_words(transcript: Transcript) -> list[Word]:
    words = [w for segment in transcript.segments for w in segment.words]
    return sorted(words, key=lambda w: w.start)


def context_around(words: list[Word], index: int, span: int = 4) -> str:
    """Texto ao redor, para o relatorio deixar o corte revisavel."""
    lo = max(0, index - span)
    hi = min(len(words), index + span + 1)
    parts = []
    for i in range(lo, hi):
        token = words[i].word.strip()
        parts.append(f"[{token}]" if i == index else token)
    return " ".join(parts)


# Abaixo disso a mediana nao descreve nada: com duas ocorrencias ela e a
# media de duas, e com uma ela e a propria palavra, o que daria razao 1.0 e
# nunca cortaria. Token com amostra menor cai no piso absoluto.
MIN_SAMPLE = 3


def token_medians(words: list[Word]) -> dict[str, float]:
    """Duracao mediana de cada token que aparece o bastante para ter mediana.

    Mediana e nao media de proposito: uma hesitacao de 0.9s no meio de cinco
    "e" de 0.2s puxaria a media para 0.34s e levantaria o piso justamente por
    causa do caso que se quer pegar.
    """
    spans: dict[str, list[float]] = {}
    for word in words:
        spans.setdefault(normalize(word.word), []).append(word.end - word.start)

    medians: dict[str, float] = {}
    for token, values in spans.items():
        if len(values) < MIN_SAMPLE:
            continue
        values.sort()
        middle = len(values) // 2
        medians[token] = (
            values[middle] if len(values) % 2
            else (values[middle - 1] + values[middle]) / 2.0
        )
    return medians


def filler_floor(
    token: str, medians: dict[str, float], rules: TrimConfig
) -> float:
    """Duracao minima para este token contar como hesitacao.

    O piso do config e absoluto e vale para todo mundo. Em cima dele, um
    token com amostra na fala ganha um piso relativo: `stretch_ratio` vezes a
    mediana dele mesmo. E isso que separa "...olha isso. E importante que" —
    verbo abrindo frase depois de uma pausa, com a duracao normal de um "e" —
    de "entao ... Eeee eu acho", que dura o dobro.

    Vale o MAIOR dos dois. O piso absoluto nunca e afrouxado por uma fala em
    que o token e naturalmente curto.
    """
    absolute = rules.filler_min_seconds
    median = medians.get(token)
    if median is None:
        return absolute
    return max(absolute, median * rules.filler_stretch_ratio)


def find_cuts(transcript: Transcript, rules: TrimConfig) -> list[Cut]:
    """Trechos a remover, em tempo da entrada original."""
    words = all_words(transcript)
    if not words:
        return []

    margin = rules.keep_margin_seconds
    wanted = {normalize(f) for f in rules.fillers}
    medians = token_medians(words)
    cuts: list[Cut] = []

    # --- hesitacao ---------------------------------------------------------
    for index, word in enumerate(words):
        token = normalize(word.word)
        if token not in wanted:
            continue

        # Na borda da gravacao o silencio e medido contra o inicio do audio,
        # nao tratado como infinito: com infinito, qualquer palavra da lista
        # que abrisse o video seria cortada.
        silence_before = word.start - (words[index - 1].end if index else 0.0)
        if silence_before < rules.filler_silence_seconds:
            continue

        # So ANTES, nao dos dois lados: ver o docstring do modulo. O intervalo
        # depois e zerado pelo transcritor, e a pausa retorica que precisa ser
        # protegida ("o problema é... que ninguem olha") tem silencio do lado
        # oposto a este, entao este teste sozinho ja a preserva.
        if word.end - word.start < filler_floor(token, medians, rules):
            continue

        cuts.append(Cut(
            start=word.start, end=word.end, reason="filler",
            token=word.word.strip(), context=context_around(words, index),
        ))

    # --- pausa -------------------------------------------------------------
    for index, (current, following) in enumerate(zip(words, words[1:])):
        gap = following.start - current.end
        if gap < rules.pause_min_seconds:
            continue
        start, end = current.end + margin, following.start - margin
        if end - start <= 0:
            continue
        cuts.append(Cut(
            start=start, end=end, reason="pause",
            context=context_around(words, index, span=3),
        ))

    return merge_cuts(cuts, rules.min_keep_seconds)


def merge_cuts(cuts: list[Cut], min_gap: float = 0.0) -> list[Cut]:
    """Funde cortes que se tocam ou quase.

    Uma hesitacao cercada de pausa gera tres cortes com lascas de audio entre
    eles, por causa da margem. Aquelas lascas seriam descartadas depois por
    serem curtas demais para manter — fundir aqui faz a lista de cortes
    descrever o que de fato acontece, em vez de tres cortes e dois sumicos.
    """
    if not cuts:
        return []
    ordered = sorted(cuts, key=lambda c: c.start)
    merged = [ordered[0]]
    for cut in ordered[1:]:
        last = merged[-1]
        if cut.start - last.end < min_gap or cut.start <= last.end:
            last.end = max(last.end, cut.end)
            if cut.reason == "filler" and last.reason == "pause":
                # a hesitacao e a informacao util para quem revisa
                last.reason, last.token, last.context = "filler", cut.token, cut.context
        else:
            merged.append(cut)
    return merged


def resolve_fps(config: Config, source_fps: float) -> float:
    """O fps da saida. `render.fps: 0` significa herdar da entrada.

    Vive aqui e nao no estagio de render porque o corte precisa do MESMO
    valor: o plano e o video renderizado tem que concordar no grid de frames.
    """
    return config.render.fps or source_fps


def keep_ranges(
    cuts: list[Cut], duration: float, rules: TrimConfig, fps: float
) -> list[KeepRange]:
    """O complemento dos cortes, encaixado no grid de frames.

    O alinhamento e obrigatorio, nao cosmetico. `atrim` corta audio com
    precisao de amostra e `trim` corta video em fronteira de frame: cortar em
    tempo arbitrario deixa cada corte com ate um frame de diferenca entre as
    duas trilhas. Medido com 20 cortes em tempos nao alinhados: 24ms de
    dessincronia acumulada. Com alinhamento: zero.
    """
    ranges: list[KeepRange] = []
    cursor = 0.0
    for cut in cuts:
        if cut.start > cursor:
            ranges.append(KeepRange(start=cursor, end=min(cut.start, duration)))
        cursor = max(cursor, cut.end)
    if cursor < duration:
        ranges.append(KeepRange(start=cursor, end=duration))

    def snap(value: float) -> float:
        return round(value * fps) / fps if fps > 0 else value

    aligned = [KeepRange(start=snap(r.start), end=snap(r.end)) for r in ranges]

    # Trecho muito curto entre dois cortes viraria um estalo. Absorve.
    return [r for r in aligned if r.duration >= rules.min_keep_seconds]


def remap_transcript(transcript: Transcript, plan: TrimPlan) -> Transcript:
    """Transcript na timeline cortada.

    Palavra que caiu inteira dentro de um corte desaparece; segmento que
    ficou sem palavra nenhuma desaparece tambem, e os indices sao
    renumerados — o estagio 3 referencia segmento por indice.
    """
    kept: list[TranscriptSegment] = []

    for segment in transcript.segments:
        words: list[Word] = []
        for word in segment.words:
            if not _inside_keep(word, plan):
                continue
            words.append(Word(
                start=round(plan.map_time(word.start), 3),
                end=round(plan.map_time(word.end), 3),
                word=word.word,
            ))

        if segment.words and not words:
            continue   # o segmento inteiro foi cortado

        start = words[0].start if words else round(plan.map_time(segment.start), 3)
        end = words[-1].end if words else round(plan.map_time(segment.end), 3)
        if end <= start:
            continue

        kept.append(TranscriptSegment(
            id=len(kept),
            start=start,
            end=end,
            text=" ".join(w.word.strip() for w in words) if words else segment.text,
            words=words,
        ))

    return Transcript(
        input_hash=transcript.input_hash,
        language=transcript.language,
        model=transcript.model,
        duration=round(sum(r.duration for r in plan.keep), 3),
        segments=kept,
    )


def _inside_keep(word: Word, plan: TrimPlan) -> bool:
    """A palavra sobrevive se o centro dela caiu num trecho mantido."""
    middle = (word.start + word.end) / 2.0
    return any(r.start <= middle <= r.end for r in plan.keep)


def build_plan(transcript: Transcript, rules: TrimConfig, fps: float) -> TrimPlan:
    key = text_hash(transcript.digest(), rules.model_dump_json(), f"{fps:.6f}")

    if not rules.enabled:
        return TrimPlan(
            input_hash=key, enabled=False,
            keep=[KeepRange(start=0.0, end=transcript.duration)],
            cuts=[],
            stats=TrimStats(
                original_seconds=transcript.duration,
                trimmed_seconds=transcript.duration,
                removed_seconds=0.0, removed_ratio=0.0,
                n_cuts=0, n_pause_cuts=0, n_filler_cuts=0,
            ),
        )

    cuts = find_cuts(transcript, rules)
    keep = keep_ranges(cuts, transcript.duration, rules, fps)
    trimmed = sum(r.duration for r in keep)
    removed = transcript.duration - trimmed

    return TrimPlan(
        input_hash=key, enabled=True, keep=keep, cuts=cuts,
        stats=TrimStats(
            original_seconds=round(transcript.duration, 3),
            trimmed_seconds=round(trimmed, 3),
            removed_seconds=round(removed, 3),
            removed_ratio=round(removed / transcript.duration, 4) if transcript.duration else 0.0,
            n_cuts=len(cuts),
            n_pause_cuts=sum(1 for c in cuts if c.reason == "pause"),
            n_filler_cuts=sum(1 for c in cuts if c.reason == "filler"),
        ),
    )
