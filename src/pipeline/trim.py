"""Deteccao de trecho morto: pausa longa e hesitacao.

A regra que mais importa aqui e negativa. Em portugues, varios sons de
hesitacao sao tambem palavras de conteudo:

    "isso E muito importante"     -> verbo, cortar destroi a frase
    "entao... E... eu acho que"   -> hesitacao, cortar melhora
    "UM problema serio"           -> artigo
    "bom, UM..., vamos ver"       -> hesitacao

Por isso a lista de `fillers` do config nunca decide sozinha. Todo candidato
precisa passar por tres testes ao mesmo tempo: estar na lista, durar acima do
minimo, e ter silencio dos DOIS lados. Palavra em fala corrida falha nos dois
ultimos; hesitacao passa nos tres.

O silencio dos dois lados, em vez de um, e o que protege a pausa retorica:
em "o problema E... que ninguem olha", a pausa depois do verbo bastaria para
marcar como hesitacao se um lado fosse suficiente.
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


def find_cuts(transcript: Transcript, rules: TrimConfig) -> list[Cut]:
    """Trechos a remover, em tempo da entrada original."""
    words = all_words(transcript)
    if not words:
        return []
    duration = transcript.duration

    margin = rules.keep_margin_seconds
    wanted = {normalize(f) for f in rules.fillers}
    cuts: list[Cut] = []

    # --- hesitacao ---------------------------------------------------------
    for index, word in enumerate(words):
        if normalize(word.word) not in wanted:
            continue
        spoken = word.end - word.start
        if spoken < rules.filler_min_seconds:
            continue

        # Na borda da gravacao o silencio e medido contra o inicio e o fim do
        # audio, nao tratado como infinito: com infinito, qualquer palavra da
        # lista que abrisse o video seria cortada.
        silence_before = word.start - (words[index - 1].end if index else 0.0)
        silence_after = (
            (words[index + 1].start if index + 1 < len(words) else duration) - word.end
        )

        # Silencio dos DOIS lados, nao de um. Com "um dos lados", a fala
        # "o problema é... que ninguem olha" perderia o verbo, porque a pausa
        # retorica depois dele bastaria para marcar como hesitacao. Exigir os
        # dois lados custa perder algum filler e protege toda frase corrida.
        if min(silence_before, silence_after) < rules.filler_silence_seconds:
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
