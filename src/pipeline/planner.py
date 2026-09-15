"""Estagio 3: o transcript vira uma EDL, via API da Anthropic.

Duas camadas de garantia sobre o retorno do modelo:

  * **structured outputs** garante que o JSON valida contra o schema — os
    campos existem e tem o tipo certo, sempre;
  * **as regras editoriais** sao semanticas e nenhum schema as expressa, entao
    passam pelo validador. EDL invalida e rejeitada e o erro especifico volta
    como feedback, ate 3 tentativas.

As fronteiras entre faixas sao indices de segmento do transcript, nao tempos.
Isso faz "nunca cortar no meio de uma frase" deixar de ser uma regra que da
para violar: a fronteira e, por construcao, fronteira de segmento.
"""

from __future__ import annotations

import anthropic

from .config import AnthropicConfig, EditorialConfig, require_key
from .edl import Budget, budget_for, build, resolve, validate
from .log import log
from .schemas import EDL, PlannedEDL, Transcript
from .util import with_retries

SYSTEM_PROMPT = """\
Voce e editor de video de um canal do YouTube cujo publico ouve mais do que
assiste: as pessoas deixam o video tocando enquanto fazem outra coisa. Sua
funcao e decidir, para cada trecho da fala, se a tela mostra a pessoa falando
(`aroll`) ou uma imagem ilustrativa (`broll`).

Porque o publico e audio-first, a imagem existe para dar descanso visual e
ancorar o que esta sendo dito — nunca para competir com a fala. Um b-roll que
troca rapido cansa mais do que ajuda.

## Como voce devolve a decisao

Uma lista de faixas contiguas, cada uma cobrindo um intervalo de indices de
segmento do transcript. As faixas precisam:

- comecar no segmento 0 e terminar no ultimo segmento, sem buraco e sem
  sobreposicao: cada faixa comeca no indice seguinte ao fim da anterior;
- alternar entre `aroll` e `broll` — duas faixas do mesmo tipo em sequencia
  sao a mesma faixa e devem ser uma so.

## Regras que nao se negociam

1. Os primeiros {intro:.0f} segundos sao `aroll`, sem excecao. O espectador
   precisa ver quem esta falando antes de qualquer coisa.
2. Cada faixa de `broll` dura no minimo {bmin:.0f} e no maximo {bmax:.0f}
   segundos.
3. No maximo {switches} trocas de tela por minuto na media do video, e nunca
   mais de {window} trocas dentro de uma janela qualquer de 60 segundos.
4. O total de `broll` fica entre {rmin:.0%} e {rmax:.0%} da duracao do video.
5. Toda faixa de `broll` carrega `concept` e `concept_tags`. Toda faixa de
   `aroll` carrega `concept` vazio e `concept_tags` vazio.

## Como escrever o `concept`

Em ingles, uma cena concreta e estatica que uma imagem unica consegue mostrar.
A imagem vai receber um movimento lento de camera, entao descreva um quadro,
nao uma acao.

- Concreto, nao abstrato: "a single wooden chair in an empty bright room",
  nao "the concept of solitude".
- Sem texto na imagem: nada de placa, titulo, grafico com rotulo, numero ou
  letreiro. O gerador escreve texto errado e a imagem fica inutilizavel.
- Sem rosto em primeiro plano. Rosto gerado disputa atencao com a pessoa do
  video e cai no vale da estranheza.
- Ligado ao que esta sendo dito naquele trecho, nao ao tema geral do video.

`concept_tags`: de 2 a 4 palavras-chave em ingles, que descrevam a cena de
forma generica o suficiente para achar algo parecido num banco de fotos.
"""


class PlanningFailed(RuntimeError):
    """O modelo nao produziu EDL valida dentro das tentativas."""


def format_transcript(transcript: Transcript) -> str:
    return "\n".join(
        f"[{s.id}] {s.start:.1f}-{s.end:.1f}s  {s.text.strip()}" for s in transcript.segments
    )


def system_prompt(rules: EditorialConfig) -> str:
    return SYSTEM_PROMPT.format(
        intro=rules.intro_aroll_seconds,
        bmin=rules.broll_min_seconds,
        bmax=rules.broll_max_seconds,
        switches=rules.max_switches_per_minute,
        window=rules.max_switches_per_minute + 1,
        rmin=rules.broll_ratio_min,
        rmax=rules.broll_ratio_max,
    )


def first_request(transcript: Transcript, budget: Budget) -> str:
    """O pedido, com as regras ja resolvidas em numeros absolutos.

    Sem isso o modelo recebe so as regras relativas ("3 trocas por minuto") e
    tem que descobrir por tentativa quantas faixas cabem. Num video curto o
    espaco de solucao pode ser um unico ponto, e as 3 tentativas se gastam
    tateando em vez de escolhendo onde cortar.
    """
    segment_avg = transcript.duration / max(1, len(transcript.segments))
    return (
        f"Duracao do video: {transcript.duration:.1f}s "
        f"({transcript.duration / 60:.1f} min), {len(transcript.segments)} segmentos "
        f"(indices 0 a {len(transcript.segments) - 1}, "
        f"~{segment_avg:.1f}s cada em media).\n\n"
        f"{budget.as_prompt()}\n\n"
        f"Transcript:\n\n{format_transcript(transcript)}"
    )


def retry_request(errors: list[str]) -> str:
    listed = "\n".join(f"- {e}" for e in errors)
    return (
        f"Essa EDL viola as regras:\n\n{listed}\n\n"
        "Corrija exatamente esses pontos e devolva a EDL inteira de novo. "
        "Mantenha o que ja estava bom."
    )


def plan(
    transcript: Transcript,
    *,
    config: AnthropicConfig,
    rules: EditorialConfig,
    client: anthropic.Anthropic | None = None,
) -> EDL:
    # Recusa de graca o video em que nenhuma EDL valida existe, antes de
    # gastar 3 tentativas de Opus descobrindo isso.
    budget = budget_for(transcript.duration, rules)
    if not budget.feasible:
        raise PlanningFailed(
            f"as regras editoriais nao tem solucao para este video: {budget.reason}.\n"
            f"Nenhuma chamada a API foi feita. Ajuste `editorial` no config — "
            f"em video curto, o caminho costuma ser baixar `intro_aroll_seconds` "
            f"e `broll_ratio_min`, ou subir `max_switches_per_minute`."
        )
    log("plan.budget", duration=f"{transcript.duration:.0f}s",
        max_switches=budget.max_switches,
        broll_spans=f"{budget.n_broll_min}-{budget.n_broll_max}",
        broll_seconds=f"{budget.broll_seconds_min:.0f}-{budget.broll_seconds_max:.0f}s")

    if client is None:
        client = anthropic.Anthropic(api_key=require_key(config.env, "planejamento editorial"))

    # O transcript e a parte grande e estavel do prompt: entre tentativas ele
    # nao muda, entao vale o breakpoint de cache.
    messages: list[dict] = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": first_request(transcript, budget),
            "cache_control": {"type": "ephemeral"},
        }],
    }]

    last_errors: list[str] = []

    for attempt in range(1, config.max_attempts + 1):
        log("plan.attempt", attempt=attempt, of=config.max_attempts, model=config.model)

        response = with_retries(
            lambda: client.with_options(timeout=config.timeout_seconds).messages.parse(
                model=config.model,
                max_tokens=config.max_tokens,
                system=[{
                    "type": "text",
                    "text": system_prompt(rules),
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=messages,
                output_format=PlannedEDL,
                output_config={"effort": config.effort},
            ),
            label=f"anthropic {config.model}",
        )

        if response.stop_reason == "refusal":
            raise PlanningFailed(
                "a API recusou a requisicao "
                f"(categoria: {getattr(response.stop_details, 'category', None)})"
            )
        if response.stop_reason == "max_tokens":
            raise PlanningFailed(
                f"a resposta foi truncada em max_tokens={config.max_tokens}; "
                "aumente anthropic.max_tokens no config"
            )

        planned: PlannedEDL = response.parsed_output
        log("plan.received", spans=len(planned.spans),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read=response.usage.cache_read_input_tokens or 0)

        try:
            segments = resolve(planned, transcript)
            errors = validate(segments, transcript, rules)
        except IndexError as exc:
            errors = [str(exc)]

        if not errors:
            edl = build(planned, transcript, model=config.model, attempts=attempt)
            log("plan.valid", broll_ratio=f"{edl.stats.broll_ratio:.0%}",
                switches_per_min=f"{edl.stats.switches_per_minute:.1f}",
                n_broll=edl.stats.n_broll, n_aroll=edl.stats.n_aroll)
            return edl

        last_errors = errors
        log("plan.rejected", attempt=attempt, violations=len(errors))
        for error in errors[:8]:
            log("plan.violation", detail=error)

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": retry_request(errors)})

    listed = "\n".join(f"  - {e}" for e in last_errors)
    raise PlanningFailed(
        f"o modelo nao produziu EDL valida em {config.max_attempts} tentativas. "
        f"Ultimas violacoes:\n{listed}"
    )
