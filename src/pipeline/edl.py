"""Resolucao e validacao da EDL contra as regras editoriais.

O modelo devolve faixas por indice de segmento do transcript; aqui elas
ganham tempo e passam pelas regras. `validate` devolve a lista de erros em
texto — e esse texto que volta para o modelo como feedback na proxima
tentativa, entao cada mensagem precisa dizer exatamente o que violou e onde.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    #
    # A comparacao e em SEGUNDOS, com a mesma tolerancia das outras regras.
    # Comparando o ratio direto, uma EDL que acerta a proporcao na mosca cai
    # em 0.49999... e e rejeitada com a mensagem "b-roll cobre 50% do video,
    # abaixo do minimo de 50%" — que o modelo nao tem como corrigir, porque
    # nao ha nada errado com ela.
    broll_seconds = sum(s.duration for s in segments if s.kind == "broll")
    ratio = broll_seconds / duration if duration else 0.0
    floor_seconds = rules.broll_ratio_min * duration
    ceil_seconds = rules.broll_ratio_max * duration

    if broll_seconds < floor_seconds - TOLERANCE:
        errors.append(
            f"b-roll cobre {broll_seconds:.1f}s de {duration:.1f}s ({ratio:.1%}), "
            f"abaixo do minimo de {floor_seconds:.1f}s ({rules.broll_ratio_min:.0%}); "
            f"faltam {floor_seconds - broll_seconds:.1f}s de b-roll"
        )
    if broll_seconds > ceil_seconds + TOLERANCE:
        errors.append(
            f"b-roll cobre {broll_seconds:.1f}s de {duration:.1f}s ({ratio:.1%}), "
            f"acima do maximo de {ceil_seconds:.1f}s ({rules.broll_ratio_max:.0%}); "
            f"devolva {broll_seconds - ceil_seconds:.1f}s para a-roll"
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


# --------------------------------------------------------------------------
# viabilidade
# --------------------------------------------------------------------------


@dataclass
class Budget:
    """O que as regras editoriais permitem para UM video especifico.

    As regras sao escritas em termos relativos ("3 trocas por minuto", "entre
    50% e 70%"). Num video de 15 min isso deixa folga; num de 2 min o espaco
    de solucao pode ser um unico ponto. Derivar os numeros absolutos antes de
    chamar a API serve para tres coisas: avisar o modelo do que ele tem (em
    vez de deixar ele descobrir por tentativa), recusar de graca um video em
    que nenhuma EDL valida existe, e nao pedir proporcao que as outras regras
    proibem.

    Esse ultimo ponto e menos obvio. Com faixas de b-roll de L segundos e
    proporcao r, a taxa de trocas e `120 * r / L` — que nao depende da duracao
    do video. Com o teto de 25s por faixa e 3 trocas por minuto, a proporcao
    maxima alcancavel e `25 * 3 / 120 = 62.5%`: o teto nominal de 70% do
    config nunca pode ser atingido. Um modelo que mire no meio da faixa
    50-70% viola o ritmo em todas as tentativas. Por isso `broll_seconds_max`
    aqui e a proporcao REAL, nao a nominal.
    """

    duration: float
    max_switches: int
    max_spans: int
    n_broll_min: int
    n_broll_max: int
    broll_seconds_min: float
    broll_seconds_max: float
    feasible: bool
    reason: str = ""

    @property
    def ratio_max(self) -> float:
        return self.broll_seconds_max / self.duration if self.duration else 0.0

    def as_prompt(self) -> str:
        if self.n_broll_min == self.n_broll_max:
            quantos = f"exatamente {self.n_broll_max} faixas de b-roll"
        else:
            quantos = f"de {self.n_broll_min} a {self.n_broll_max} faixas de b-roll"
        por_faixa = self.broll_seconds_min / max(1, self.n_broll_max)
        return (
            f"Para ESTE video, as regras acima ja resolvidas em numeros. "
            f"Estes valores tem precedencia sobre as faixas relativas das "
            f"regras gerais:\n"
            f"- no maximo {self.max_switches} trocas de tela no total, "
            f"ou seja no maximo {self.max_spans} faixas somando a-roll e b-roll\n"
            f"- {quantos}\n"
            f"- b-roll somando entre {self.broll_seconds_min:.1f}s e "
            f"{self.broll_seconds_max:.1f}s (de {self.ratio_max:.0%} para baixo; "
            f"o limite de trocas impede mais que isso, independente do teto "
            f"nominal)\n"
            f"- com {self.n_broll_max} faixas de b-roll, cada uma precisa de "
            f"pelo menos {por_faixa:.1f}s para o total fechar\n"
            f"Mire perto do MEIO dessa janela de segundos, nao no topo: sobrar "
            f"margem no ritmo e o que permite encaixar os cortes em fronteira "
            f"de segmento do transcript."
        )


def budget_for(duration: float, rules: EditorialConfig) -> Budget:
    """Traduz as regras relativas nos limites absolutos deste video."""
    max_switches = int(rules.max_switches_per_minute * duration / 60.0)
    max_spans = max_switches + 1
    seconds_min = rules.broll_ratio_min * duration
    seconds_max = rules.broll_ratio_max * duration

    # A primeira faixa e sempre a-roll (regra da intro) e as faixas alternam,
    # entao S faixas dao S//2 de b-roll.
    n_broll_ceiling = max_spans // 2

    # quantas faixas de b-roll conseguem cobrir o minimo exigido
    viable = [
        n for n in range(1, n_broll_ceiling + 1)
        if n * rules.broll_max_seconds >= seconds_min - TOLERANCE
        and n * rules.broll_min_seconds <= seconds_max + TOLERANCE
    ]

    # O teto efetivo: mais que `n_broll_max` faixas de `broll_max_seconds`
    # nao cabe, entao pedir a proporcao nominal seria pedir o impossivel.
    reachable_max = (max(viable) if viable else 0) * rules.broll_max_seconds
    budget = Budget(
        duration=duration,
        max_switches=max_switches,
        max_spans=max_spans,
        n_broll_min=min(viable) if viable else 0,
        n_broll_max=max(viable) if viable else 0,
        broll_seconds_min=seconds_min,
        broll_seconds_max=min(seconds_max, reachable_max),
        feasible=bool(viable),
    )

    if not viable:
        if n_broll_ceiling == 0:
            budget.reason = (
                f"o video tem {duration:.0f}s, e com no maximo "
                f"{rules.max_switches_per_minute} trocas por minuto nao cabe nem "
                f"uma faixa de b-roll"
            )
        else:
            budget.reason = (
                f"o video tem {duration:.0f}s e comporta no maximo "
                f"{n_broll_ceiling} faixa(s) de b-roll (limite de "
                f"{rules.max_switches_per_minute} trocas/min). Mesmo com "
                f"{rules.broll_max_seconds:.0f}s cada, isso da "
                f"{n_broll_ceiling * rules.broll_max_seconds:.0f}s, abaixo dos "
                f"{seconds_min:.0f}s que o minimo de "
                f"{rules.broll_ratio_min:.0%} exige"
            )

    # a intro obrigatoria tem que caber no a-roll que sobra
    aroll_min = duration - seconds_max
    if budget.feasible and aroll_min < rules.intro_aroll_seconds - TOLERANCE:
        budget.feasible = False
        budget.reason = (
            f"com b-roll em {rules.broll_ratio_max:.0%} sobram {aroll_min:.0f}s "
            f"de a-roll, menos que os {rules.intro_aroll_seconds:.0f}s de intro "
            f"obrigatoria"
        )

    return budget
