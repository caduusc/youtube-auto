"""Estagio 1: da ideia para o roteiro.

O roteiro e o artefato de maior alavancagem do pipeline inteiro: tudo depois
dele decorre dele — quais imagens existem, onde entram, o que o video defende.
E uma chamada por video, entao nao ha razao para economizar modelo aqui.

O modelo devolve ESTRUTURA e nao Markdown pronto. Assim a validacao sai de
graca e o `script.render` continua sendo o unico lugar que sabe escrever o
formato do arquivo.
"""

from __future__ import annotations

import anthropic

from .config import AnthropicConfig, ScriptConfig
from .log import log
from .schemas import PlannedScript, Script
from .util import with_retries

SYSTEM_PROMPT = """\
Voce escreve roteiro para um canal do YouTube cujo publico OUVE mais do que
assiste: as pessoas deixam o video tocando enquanto fazem outra coisa.

Isso muda como o roteiro tem que ser escrito. O espectador nao esta olhando,
entao nada pode depender de ver a tela: nenhum "como voces estao vendo aqui",
nenhuma referencia a algo que so existe na imagem. O que a imagem faz e dar
descanso visual e ancorar o que esta sendo dito — ela nunca carrega
informacao que a fala nao carregue.

## O que voce devolve

O roteiro dividido em BEATS. Um beat e um trecho em que uma ideia e
desenvolvida — nao um paragrafo por tamanho, mas uma unidade de sentido. Cada
beat vira um pedaco do arquivo que a pessoa vai editar a mao, e mais tarde as
imagens sao ancoradas em trechos dele.

Beat tipico: 30 a 90 segundos falados. Se um beat ficou com mais de tres
ideias dentro, ele e dois beats.

## Como escrever a fala

Escreva para ser DITO, nao lido. A pessoa vai gravar isso falando, entao:

- frase curta, uma ideia por frase;
- sem construcao que so funciona no papel (aposto longo, parentese, lista com
  travessao no meio da frase);
- sem numero que a pessoa vai tropecar ao falar: "quase duzentos reais", nao
  "R$ 197,43";
- sem jargao que ela mesma nao usaria na conversa.

Nao escreva indicacao de cena, de imagem, nem de corte. Isso e o estagio
seguinte, e misturar os dois aqui piora os dois.

## Duracao

O alvo e {target:.0f} segundos falados, o que a {wpm:.0f} palavras por minuto
da aproximadamente {words:.0f} palavras no total. Fique entre {low:.0f} e
{high:.0f} palavras. Nao encha linguica para chegar no alvo: se a ideia acaba
antes, entregue mais curto e diga isso no `argument`.

## Estrutura

O roteiro precisa defender UM argumento, que e o que voce escreve em
`argument`. Cada beat empurra aquele argumento adiante ou sustenta ele com
prova, exemplo ou historia. Beat que nao faz nem uma coisa nem outra sai.

O primeiro beat abre sem enrolacao — nada de "oi pessoal, tudo bem, hoje eu
vou falar sobre". Comece pelo que interessa.
"""


class WritingFailed(RuntimeError):
    pass


def system_prompt(target_seconds: float, rules: ScriptConfig) -> str:
    words = target_seconds / 60.0 * rules.words_per_minute
    return SYSTEM_PROMPT.format(
        target=target_seconds, wpm=rules.words_per_minute, words=words,
        low=words * 0.85, high=words * 1.15,
    )


def validate(planned: PlannedScript) -> list[str]:
    """Violacoes em portugues, para voltarem como feedback ao modelo."""
    problemas: list[str] = []

    if not planned.beats:
        return ["o roteiro nao tem beat nenhum"]

    ids = [b.id for b in planned.beats]
    if ids != list(range(1, len(ids) + 1)):
        problemas.append(
            f"os ids dos beats precisam comecar em 1 e seguir sem buraco; "
            f"voce mandou {ids}"
        )

    for beat in planned.beats:
        if not beat.text.strip():
            problemas.append(f"o beat {beat.id} esta sem texto")

    for campo in ("subject", "argument", "viewer_takeaway"):
        if not getattr(planned, campo).strip():
            problemas.append(f"o campo {campo} esta vazio")

    return problemas


def write(
    idea: str,
    *,
    slug: str,
    target_seconds: float,
    config: AnthropicConfig,
    rules: ScriptConfig,
    client: anthropic.Anthropic,
) -> Script:
    """A ideia dele vira roteiro. Tenta de novo com o erro como feedback."""
    messages: list[dict] = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": f"A ideia do video, nas palavras de quem vai gravar:\n\n{idea}",
            "cache_control": {"type": "ephemeral"},
        }],
    }]

    for attempt in range(1, config.max_attempts + 1):
        log("script.attempt", attempt=attempt, of=config.max_attempts, model=config.model)

        response = with_retries(
            lambda: client.with_options(timeout=config.timeout_seconds).messages.parse(
                model=config.model,
                max_tokens=config.max_tokens,
                system=[{
                    "type": "text",
                    "text": system_prompt(target_seconds, rules),
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=messages,
                output_format=PlannedScript,
                output_config={"effort": config.effort},
            ),
            label=f"anthropic roteiro {config.model}",
        )

        if response.stop_reason == "refusal":
            raise WritingFailed(
                "a API recusou escrever o roteiro "
                f"(categoria: {getattr(response.stop_details, 'category', None)})"
            )
        if response.stop_reason == "max_tokens":
            raise WritingFailed(
                f"o roteiro foi truncado em max_tokens={config.max_tokens}; "
                "aumente anthropic.max_tokens no config"
            )

        planned: PlannedScript = response.parsed_output
        problemas = validate(planned)

        if not problemas:
            script = Script(slug=slug, duration_target_seconds=target_seconds,
                            **planned.model_dump())
            log("script.ok", beats=len(script.beats),
                palavras=sum(len(b.text.split()) for b in script.beats),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens)
            return script

        log("script.invalid", attempt=attempt, problemas=len(problemas))
        for problema in problemas:
            log("script.problema", detail=problema)

        messages += [
            {"role": "assistant", "content": planned.model_dump_json()},
            {"role": "user", "content":
                "O roteiro nao passou na validacao:\n\n"
                + "\n".join(f"- {p}" for p in problemas)
                + "\n\nCorrija e devolva o roteiro inteiro."},
        ]

    raise WritingFailed(
        f"o roteiro nao ficou valido em {config.max_attempts} tentativas. "
        f"Ultimo problema: {problemas[0] if problemas else 'desconhecido'}"
    )
