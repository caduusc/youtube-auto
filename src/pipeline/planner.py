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
from .schemas import EDL, Brief, PlannedEDL, Transcript
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

Voce recebeu um briefing visual do video. Use o `visual_vocabulary` dele como
paleta: as cenas devem sair daquele repertorio, para o conjunto ter unidade.
Nao repita elemento em dois b-rolls seguidos, e nao use nada do `avoid`.

O briefing define o VOCABULARIO; cada cena continua ancorada ao proprio
trecho. Uma imagem que ilustra o tema geral do video mas nao o que esta sendo
dito naquele momento e pior que uma imagem generica: o espectador sente o
descolamento entre o que ouve e o que ve.

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


BRIEF_PROMPT = """\
Voce vai ler a transcricao inteira de um video e devolver um briefing visual.
Nao decida cortes nem tempos aqui: isso vem depois, numa segunda etapa.

O objetivo do briefing e dar UNIDADE as imagens do video. Sem ele, cada trecho
recebe uma imagem plausivel isoladamente e o conjunto parece um banco de fotos
sorteado.

## O que preencher

- `subject` e `argument`: do que o video trata e o que ele defende. Seja
  especifico ao video, nao ao tema. "produtividade" nao serve; "por que
  metodo de produtividade falha quando a pessoa nao tem clareza do objetivo"
  serve.
- `audience_takeaway`: o que a pessoa leva embora depois de ouvir.
- `visual_vocabulary`: de 6 a 10 elementos visuais CONCRETOS em ingles que
  possam atravessar o video inteiro — objetos, ambientes, materiais, tipo de
  luz. Eles sao o que faz imagens geradas em chamadas independentes parecerem
  do mesmo video. Prefira o concreto e o duradouro ("a worn leather notebook
  on a wooden table") ao abstrato e ao datado ("futuristic technology").
- `avoid`: de 2 a 5 clices visuais em ingles a evitar NESTE video, por serem
  obvios ou nao dizerem nada. Se o video fala de decisao, "chessboard" e
  provavelmente um desses.

Nada de texto, letreiro ou rosto em primeiro plano em nenhuma sugestao.
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


def write_brief(
    transcript: Transcript, *, config: AnthropicConfig, client: anthropic.Anthropic
) -> Brief:
    """Fase 1: le a transcricao inteira e devolve o briefing visual."""
    response = with_retries(
        lambda: client.with_options(timeout=config.timeout_seconds).messages.parse(
            model=config.model,
            max_tokens=config.max_tokens,
            system=[{"type": "text", "text": BRIEF_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": [{
                "type": "text",
                "text": f"Transcricao completa:\n\n{format_transcript(transcript)}",
                "cache_control": {"type": "ephemeral"},
            }]}],
            output_format=Brief,
            output_config={"effort": config.effort},
        ),
        label=f"anthropic brief {config.model}",
    )

    if response.stop_reason == "refusal":
        raise PlanningFailed(
            "a API recusou o briefing "
            f"(categoria: {getattr(response.stop_details, 'category', None)})"
        )
    if response.stop_reason == "max_tokens":
        raise PlanningFailed(
            f"o briefing foi truncado em max_tokens={config.max_tokens}"
        )

    brief: Brief = response.parsed_output
    log("plan.brief", subject=brief.subject[:70],
        vocabulary=len(brief.visual_vocabulary), avoid=len(brief.avoid),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens)
    return brief


def brief_block(brief: Brief) -> str:
    """O briefing como texto, para entrar no pedido da fase 2."""
    return (
        "Briefing visual deste video:\n"
        f"- assunto: {brief.subject}\n"
        f"- argumento: {brief.argument}\n"
        f"- o que o espectador leva: {brief.audience_takeaway}\n"
        f"- vocabulario visual (use como paleta): "
        f"{', '.join(brief.visual_vocabulary)}\n"
        f"- evitar neste video: {', '.join(brief.avoid)}"
    )


def first_request(transcript: Transcript, budget: Budget, brief: Brief | None = None) -> str:
    """O pedido, com as regras ja resolvidas em numeros absolutos.

    Sem isso o modelo recebe so as regras relativas ("3 trocas por minuto") e
    tem que descobrir por tentativa quantas faixas cabem. Num video curto o
    espaco de solucao pode ser um unico ponto, e as 3 tentativas se gastam
    tateando em vez de escolhendo onde cortar.
    """
    segment_avg = transcript.duration / max(1, len(transcript.segments))
    header = (
        f"Duracao do video: {transcript.duration:.1f}s "
        f"({transcript.duration / 60:.1f} min), {len(transcript.segments)} segmentos "
        f"(indices 0 a {len(transcript.segments) - 1}, "
        f"~{segment_avg:.1f}s cada em media)."
    )
    blocks = [header]
    if brief is not None:
        blocks.append(brief_block(brief))
    blocks.append(budget.as_prompt())
    blocks.append(f"Transcript:\n\n{format_transcript(transcript)}")
    return "\n\n".join(blocks)


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
    two_phase: bool = True,
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

    brief = write_brief(transcript, config=config, client=client) if two_phase else None

    # O transcript e a parte grande e estavel do prompt: entre tentativas ele
    # nao muda, entao vale o breakpoint de cache.
    messages: list[dict] = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": first_request(transcript, budget, brief),
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
            edl.brief = brief
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
