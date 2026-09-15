"""Resolucao e validacao da EDL contra as regras editoriais.

O modelo devolve faixas por indice de segmento do transcript; aqui elas
ganham tempo e passam pelas regras. `validate` devolve a lista de erros em
texto — e esse texto que volta para o modelo como feedback na proxima
tentativa, entao cada mensagem precisa dizer exatamente o que violou e onde.
"""

from __future__ import annotations

from .config import EditorialConfig
from .schemas import EDL, EDLSegment, EDLStats, PlannedEDL, Transcript

TOLERANCE = 0.05


def resolve(planned: PlannedEDL, transcript: Transcript) -> list[EDLSegment]:
    """Converte faixas de indice em segmentos com tempo.

    Nao valida nada: faixas incoerentes viram erro em `validate`, com
    mensagem util. So o que impede de calcular tempo levanta aqui.
    """
    n = len(transcript.segments)
    resolved: list[EDLSegment] = []
    for span in planned.spans:
        if not (0 <= span.first_segment < n) or not (0 <= span.last_segment < n):
            raise IndexError(
                f"faixa {span.kind} referencia segmentos {span.first_segment}-{span.last_segment}, "
                f"mas o transcript tem indices de 0 a {n - 1}"
            )
        if span.last_segment < span.first_segment:
            raise IndexError(
                f"faixa {span.kind} tem last_segment ({span.last_segment}) "
                f"antes de first_segment ({span.first_segment})"
            )
        resolved.append(
            EDLSegment(
                **span.model_dump(),
                start=transcript.segments[span.first_segment].start,
                end=transcript.segments[span.last_segment].end,
            )
        )
    return resolved


def _worst_window(segments: list[EDLSegment]) -> tuple[float, float]:
    """Pior janela deslizante de 60s. Devolve (contagem, inicio da janela).

    Contar por minuto de relogio deixaria passar 6 trocas em 10 segundos se
    elas caissem em cima da fronteira do minuto.
    """
    boundaries = [s.start for s in segments[1:]]
    worst, worst_at = 0.0, 0.0
    for i, origin in enumerate(boundaries):
        count = sum(1 for b in boundaries[i:] if b < origin + 60.0)
        if count > worst:
            worst, worst_at = float(count), origin
    return worst, worst_at


def _switch_rate(segments: list[EDLSegment], duration: float) -> float:
    """Trocas de tela por minuto no video inteiro."""
    if duration <= 0:
        return 0.0
    return len(segments[1:]) / (duration / 60.0)


def stats(segments: list[EDLSegment], duration: float) -> EDLStats:
    broll_seconds = sum(s.duration for s in segments if s.kind == "broll")
    worst, _ = _worst_window(segments)
    return EDLStats(
        broll_ratio=broll_seconds / duration if duration else 0.0,
        switches_per_minute=_switch_rate(segments, duration),
        switches_per_minute_max=worst,
        n_aroll=sum(1 for s in segments if s.kind == "aroll"),
        n_broll=sum(1 for s in segments if s.kind == "broll"),
        broll_seconds=broll_seconds,
    )


def validate(
    segments: list[EDLSegment],
    transcript: Transcript,
    rules: EditorialConfig,
) -> list[str]:
    """Devolve [] se a EDL respeita todas as regras editoriais."""
    errors: list[str] = []
    duration = transcript.duration

    if not segments:
        return ["a EDL nao tem nenhuma faixa"]

    # --- cobertura: a timeline inteira, sem buraco e sem sobreposicao ------
    if segments[0].first_segment != 0:
        errors.append(
            f"a primeira faixa comeca no segmento {segments[0].first_segment}; "
            "precisa comecar no segmento 0"
        )
    last_index = len(transcript.segments) - 1
    if segments[-1].last_segment != last_index:
        errors.append(
            f"a ultima faixa termina no segmento {segments[-1].last_segment}; "
            f"precisa terminar no segmento {last_index} para cobrir o video inteiro"
        )
    for prev, nxt in zip(segments, segments[1:]):
        if nxt.first_segment != prev.last_segment + 1:
            errors.append(
                f"as faixas nao encaixam: uma termina no segmento {prev.last_segment} "
                f"e a proxima comeca no {nxt.first_segment}; a proxima precisa comecar "
                f"no {prev.last_segment + 1}"
            )
        if prev.kind == nxt.kind:
            errors.append(
                f"duas faixas {prev.kind} consecutivas em {prev.start:.1f}s e "
                f"{nxt.start:.1f}s; junte as duas numa so"
            )

    # --- intro sempre a-roll ----------------------------------------------
    intro = rules.intro_aroll_seconds
    for seg in segments:
        if seg.kind == "broll" and seg.start < intro - TOLERANCE:
            errors.append(
                f"b-roll comeca em {seg.start:.1f}s, dentro dos primeiros "
                f"{intro:.0f}s que sao obrigatoriamente a-roll"
            )

    # --- duracao de cada b-roll -------------------------------------------
    for seg in segments:
        if seg.kind != "broll":
            continue
        if seg.duration < rules.broll_min_seconds - TOLERANCE:
            errors.append(
                f"o b-roll em {seg.start:.1f}s dura {seg.duration:.1f}s, abaixo do "
                f"minimo de {rules.broll_min_seconds:.0f}s; estenda ou vire a-roll"
            )
        if seg.duration > rules.broll_max_seconds + TOLERANCE:
            errors.append(
                f"o b-roll em {seg.start:.1f}s dura {seg.duration:.1f}s, acima do "
                f"maximo de {rules.broll_max_seconds:.0f}s; quebre em dois ou encurte"
            )

    # --- ritmo -------------------------------------------------------------
    #
    # "no maximo 3 trocas por minuto" nao pode ser lido como janela
    # deslizante estrita: b-roll de no maximo 25s cobrindo pelo menos 50% do
    # video forca um ciclo de no maximo 50s, e qualquer padrao com ciclo
    # abaixo de 60s tem alguma janela de 60s com 4 fronteiras dentro. As duas
    # regras juntas seriam inviaveis.
    #
    # Entao a taxa vale no video inteiro, e a janela ganha a folga de 1 que
    # o efeito de borda produz. Isso ainda barra rajada local: 8s de b-roll
    # picado com a-roll curto estoura as duas contas.
    rate = _switch_rate(segments, duration)
    if rate > rules.max_switches_per_minute + TOLERANCE:
        errors.append(
            f"{rate:.1f} trocas de tela por minuto no video inteiro; "
            f"o maximo e {rules.max_switches_per_minute}"
        )
    window_cap = rules.max_switches_per_minute + 1
    worst, worst_at = _worst_window(segments)
    if worst > window_cap:
        errors.append(
            f"{worst:.0f} trocas de tela nos 60s a partir de {worst_at:.1f}s; "
            f"o maximo em qualquer janela de 60s e {window_cap}"
        )

    # --- proporcao de b-roll ----------------------------------------------
    broll_seconds = sum(s.duration for s in segments if s.kind == "broll")
    ratio = broll_seconds / duration if duration else 0.0
    if ratio < rules.broll_ratio_min:
        errors.append(
            f"b-roll cobre {ratio:.0%} do video, abaixo do minimo de "
            f"{rules.broll_ratio_min:.0%}; converta mais trechos de a-roll em b-roll"
        )
    if ratio > rules.broll_ratio_max:
        errors.append(
            f"b-roll cobre {ratio:.0%} do video, acima do maximo de "
            f"{rules.broll_ratio_max:.0%}; devolva trechos para a-roll"
        )

    # --- conteudo de cada b-roll ------------------------------------------
    for seg in segments:
        if seg.kind == "broll":
            if not seg.concept.strip():
                errors.append(f"o b-roll em {seg.start:.1f}s esta sem `concept`")
            n_tags = len(seg.concept_tags)
            if not (rules.concept_tags_min <= n_tags <= rules.concept_tags_max):
                errors.append(
                    f"o b-roll em {seg.start:.1f}s tem {n_tags} concept_tags; "
                    f"precisa de {rules.concept_tags_min} a {rules.concept_tags_max}"
                )
        else:
            if seg.concept.strip() or seg.concept_tags:
                errors.append(
                    f"a faixa de a-roll em {seg.start:.1f}s trouxe concept/concept_tags; "
                    "so b-roll tem esses campos"
                )

    return errors


def build(planned: PlannedEDL, transcript: Transcript, model: str, attempts: int) -> EDL:
    segments = resolve(planned, transcript)
    return EDL(
        input_hash=transcript.digest(),
        model=model,
        attempts=attempts,
        duration=transcript.duration,
        segments=segments,
        stats=stats(segments, transcript.duration),
    )
