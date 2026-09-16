"""Estagio 2: do roteiro para os beats visuais.

O que este estagio NAO faz e o que mais importa: ele nao escreve segundo
absoluto nenhum. Ele escreve, para cada imagem, um trecho literal do roteiro
onde ela entra — o `script_anchor` — e o estagio de alinhamento descobre depois
onde aquele trecho caiu na fala real.

A razao e que o roteiro nao vai ser seguido palavra por palavra. Pedir ao
modelo um tempo absoluto seria pedir que ele previsse a entrega; qualquer
improviso de oito segundos deslocaria todas as imagens seguintes. Com ancora
de texto, improviso nao desalinha nada.
"""

from __future__ import annotations

import re
import unicodedata

import anthropic

from .config import AnthropicConfig, EditorialConfig, RenderConfig
from .log import log
from .util import with_retries
from .schemas import PlannedBeat, PlannedStoryboard, Script, Storyboard, StoryboardBeat

# Mesma tolerancia do `edl.py`: comparacao de segundos com folga, para uma
# proposta que acerta a proporcao na mosca nao cair em 0.4999 e ser rejeitada
# com uma mensagem que o modelo nao tem como corrigir.
TOLERANCE = 0.05


def normalize(text: str) -> str:
    """Para casar ancora com roteiro sem exigir pontuacao identica.

    O modelo copia o trecho, mas copia com a virgula de fora, ou com reticencia
    no lugar de ponto, ou com espaco duplo. Nenhuma dessas diferencas muda ONDE
    a imagem entra, e rejeitar por causa delas gastaria tentativa de API num
    problema que nao e editorial.
    """
    sem_acento = "".join(
        ch for ch in unicodedata.normalize("NFKD", text.lower())
        if not unicodedata.combining(ch)
    )
    return re.sub(r"[^a-z0-9 ]+", " ", sem_acento).strip()


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def find_anchor(anchor: str, beat_text: str) -> int:
    """Onde o ancora comeca no texto do beat, em palavras. -1 se nao achar.

    A posicao e em PALAVRAS e nao em caracteres porque e ela que vai ser
    comparada com o transcript, que tambem e uma sequencia de palavras.

    A busca e por sequencia de palavras e nao por substring: "ostrar como"
    casaria dentro de "mostrar como" e devolveria uma posicao deslocada, o que
    passaria pela validacao e alinharia a imagem no lugar errado.
    """
    alvo = collapse(normalize(anchor)).split()
    if not alvo:
        return -1
    corpo = collapse(normalize(beat_text)).split()
    for inicio in range(len(corpo) - len(alvo) + 1):
        if corpo[inicio:inicio + len(alvo)] == alvo:
            return inicio
    return -1


def validate(
    planned: list[PlannedBeat],
    script: Script,
    rules: EditorialConfig,
    render: RenderConfig,
    spoken_seconds: float,
) -> list[str]:
    """Violacoes em portugues, para voltarem como feedback ao modelo.

    Mesma forma do `edl.validate`: lista de frases que dizem o que esta errado
    E o que fazer. "beat 7 invalido" faz o modelo tentar de novo no escuro;
    "o beat 7 nao existe no roteiro, que tem beats 1 a 5" ele conserta.
    """
    problemas: list[str] = []

    if not planned:
        return ["o storyboard nao tem beat nenhum"]

    ids_validos = {b.id for b in script.beats}

    # --- cada beat, por si ------------------------------------------------
    for item in planned:
        beat = script.beat(item.beat_id)
        if beat is None:
            problemas.append(
                f"o beat {item.beat_id} nao existe no roteiro, que tem os beats "
                f"{sorted(ids_validos)}"
            )
            continue

        if find_anchor(item.script_anchor, beat.text) < 0:
            problemas.append(
                f"o ancora do beat {item.beat_id} nao aparece no texto dele. "
                f"Copie um trecho LITERAL do roteiro; voce escreveu "
                f'"{item.script_anchor[:60]}"'
            )

        if not 1 <= item.sub_shots <= render.max_sub_shots:
            problemas.append(
                f"o beat {item.beat_id} pede {item.sub_shots} sub-planos; o limite "
                f"e 1 a {render.max_sub_shots} (e geometrico: recorte menor que um "
                f"quadrante da imagem nao cabe em {render.width}x{render.height} "
                f"sem ampliar)"
            )

        n_tags = len(item.concept_tags)
        if not rules.concept_tags_min <= n_tags <= rules.concept_tags_max:
            problemas.append(
                f"o beat {item.beat_id} tem {n_tags} concept_tags; o esperado e "
                f"{rules.concept_tags_min} a {rules.concept_tags_max}"
            )

        if not item.rationale.strip():
            problemas.append(
                f"o beat {item.beat_id} esta sem rationale — e o que se le na "
                f"revisao para decidir se aprova"
            )

    # --- varios beats visuais no mesmo beat do roteiro --------------------
    # Permitido: um beat longo pode pedir duas imagens. Mas as ancoras tem que
    # estar em ordem, senao duas imagens disputam o mesmo momento da fala.
    por_beat: dict[int, list[PlannedBeat]] = {}
    for item in planned:
        por_beat.setdefault(item.beat_id, []).append(item)

    for beat_id, itens in por_beat.items():
        beat = script.beat(beat_id)
        if beat is None or len(itens) < 2:
            continue
        posicoes = [find_anchor(i.script_anchor, beat.text) for i in itens]
        if any(p < 0 for p in posicoes):
            continue   # ja reportado acima
        if posicoes != sorted(posicoes) or len(set(posicoes)) != len(posicoes):
            problemas.append(
                f"o beat {beat_id} tem {len(itens)} imagens, mas as ancoras nao "
                f"estao em ordem crescente no texto. Duas imagens no mesmo ponto "
                f"da fala disputam a tela"
            )

    # --- cobertura --------------------------------------------------------
    # Antes de gravar nao ha transcript, entao a duracao vem da estimativa de
    # fala do roteiro. A comparacao e em SEGUNDOS e com tolerancia, pelo mesmo
    # motivo do edl.py.
    if spoken_seconds > 0:
        tela = sum(
            i.sub_shots * render.sub_shot_seconds
            for i in planned
            if 1 <= i.sub_shots <= render.max_sub_shots
        )
        piso = spoken_seconds * rules.broll_ratio_min
        teto = spoken_seconds * rules.broll_ratio_max
        if tela < piso - TOLERANCE:
            faltam = piso - tela
            problemas.append(
                f"as imagens cobrem {tela:.0f}s de {spoken_seconds:.0f}s falados "
                f"({tela/spoken_seconds:.0%}), abaixo do minimo de {piso:.0f}s "
                f"({rules.broll_ratio_min:.0%}); faltam {faltam:.0f}s — adicione "
                f"beats visuais ou suba sub_shots nos que ja existem"
            )
        elif tela > teto + TOLERANCE:
            problemas.append(
                f"as imagens cobrem {tela:.0f}s de {spoken_seconds:.0f}s falados "
                f"({tela/spoken_seconds:.0%}), acima do maximo de {teto:.0f}s "
                f"({rules.broll_ratio_max:.0%}); tire beats ou baixe sub_shots"
            )

    return problemas


def resolve(
    planned: list[PlannedBeat], script: Script, render: RenderConfig, input_hash: str
) -> Storyboard:
    """Os beats validados, com o que o pipeline derivou deles."""
    beats: list[StoryboardBeat] = []
    for item in planned:
        beat = script.beat(item.beat_id)
        offset = find_anchor(item.script_anchor, beat.text) if beat else -1
        beats.append(StoryboardBeat(
            **item.model_dump(),
            anchor_offset=offset,
            seconds_per_shot=render.sub_shot_seconds,
        ))
    beats.sort(key=lambda b: (b.beat_id, b.anchor_offset))
    return Storyboard(input_hash=input_hash, beats=beats)


def render_text(storyboard: Storyboard, script: Script) -> str:
    """O storyboard para revisar no terminal, antes de gerar imagem.

    Mostra o trecho do roteiro ao lado do conceito: sem isso voce aprova uma
    lista de descricoes de imagem sem saber sobre o que cada uma entra.
    """
    linhas = ["", "  Storyboard", "  " + "-" * 72]
    for beat in storyboard.beats:
        do_roteiro = script.beat(beat.beat_id)
        titulo = f" — {do_roteiro.title}" if do_roteiro and do_roteiro.title else ""
        linhas.append(
            f"  beat {beat.beat_id}{titulo}   "
            f"{beat.sub_shots} plano{'s' if beat.sub_shots != 1 else ''} "
            f"de {beat.seconds_per_shot:g}s = {beat.screen_seconds:g}s na tela"
        )
        linhas.append(f'    ancora:  "{beat.script_anchor}"')
        linhas.append(f"    imagem:  {beat.concept}")
        linhas.append(f"    tags:    {', '.join(beat.concept_tags)}")
        linhas.append(f"    porque:  {beat.rationale}")
        linhas.append("")
    total = storyboard.total_screen_seconds
    linhas += [
        "  " + "-" * 72,
        f"  {storyboard.n_images} "
        f"{'imagens' if storyboard.n_images != 1 else 'imagem'} "
        f"({sum(b.sub_shots for b in storyboard.beats)} planos), "
        f"{total:.0f}s na tela",
        "",
    ]
    return "\n".join(linhas)


# --------------------------------------------------------------------------
# o agente
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Voce e diretor de fotografia de um canal do YouTube cujo publico OUVE mais do
que assiste. Voce recebe um roteiro aprovado e decide quais imagens entram e
em que ponto da fala cada uma entra.

Porque o publico e audio-first, a imagem existe para dar descanso visual e
ancorar o que esta sendo dito — nunca para competir com a fala nem para
carregar informacao que a fala nao carrega.

## Voce NAO escreve tempo

Nada de segundo, minuto ou timestamp. Para cada imagem voce copia um
`script_anchor`: um trecho CURTO E LITERAL do texto daquele beat, exatamente
como esta escrito no roteiro, marcando onde a imagem entra.

A razao e que o roteiro nao vai ser lido palavra por palavra. Quem gravar vai
improvisar, e o tempo real de cada trecho so existe depois da gravacao. O
ancora e o que faz a imagem achar o lugar dela sozinha.

Copie o trecho, nao parafraseie. Tres a oito palavras basta, e tem que
aparecer identico no beat que voce indicou em `beat_id`.

## Sub-planos: uma imagem, varios planos

Cada imagem rende de 1 a {max_sub} planos, recortando regioes diferentes dela
mesma — um plano aberto, um detalhe, outro detalhe. Cada plano fica
{shot_seconds:g} segundos na tela.

Entao `sub_shots: 4` significa uma imagem gerada ocupando
{four_shots:g} segundos de video. Use mais sub-planos onde o trecho e mais
longo e a ideia sustenta o olhar; use 1 onde a imagem e uma pontuacao rapida.

Pense na imagem como um quadro que a camera vai percorrer, nao como uma
ilustracao para ser vista de uma vez.

## Quanto cobrir

O roteiro tem aproximadamente {spoken:.0f} segundos falados. O total de tela
com imagem fica entre {low:.0f} e {high:.0f} segundos, ou seja entre
{rmin:.0%} e {rmax:.0%} do video. Distribua: nao concentre tudo no comeco nem
deixe tres beats seguidos sem nenhuma imagem.

Os primeiros {intro:.0f} segundos do video sao a pessoa falando, sem imagem —
o espectador precisa ver quem esta falando antes de qualquer coisa. Na
pratica, isso quer dizer que o primeiro beat costuma ficar sem imagem.

## Como escrever o `concept`

Em ingles, uma cena concreta e estatica que uma imagem unica consegue mostrar.
A imagem vai receber movimento lento de camera e ser recortada em sub-planos,
entao descreva um QUADRO com mais de um ponto de interesse — nao uma acao, e
nao um objeto solitario no centro.

- Concreto, nao abstrato: "a single wooden chair in an empty bright room",
  nao "the concept of solitude".
- Sem texto na imagem: nada de placa, titulo, grafico com rotulo, numero ou
  letreiro. O gerador escreve texto errado e a imagem fica inutilizavel.
- Sem rosto em primeiro plano. Rosto gerado disputa atencao com a pessoa do
  video e cai no vale da estranheza.
- Ligado ao que esta sendo dito NAQUELE trecho, nao ao tema geral do video.
  Uma imagem que ilustra o assunto mas nao o que esta sendo falado naquele
  momento e pior que uma imagem generica: o espectador sente o descolamento
  entre o que ouve e o que ve.
- Sem repetir o mesmo elemento em duas imagens seguidas.

`concept_tags`: de {tags_min} a {tags_max} palavras-chave em ingles, genericas
o bastante para achar algo parecido num banco de fotos.

`rationale`: uma frase em portugues dizendo por que esta imagem, neste ponto.
E o que a pessoa le na revisao para decidir se aprova — escreva para ela, nao
para o sistema.
"""


class StoryboardFailed(RuntimeError):
    pass


def system_prompt(
    rules: EditorialConfig, render: RenderConfig, spoken_seconds: float
) -> str:
    return SYSTEM_PROMPT.format(
        max_sub=render.max_sub_shots,
        shot_seconds=render.sub_shot_seconds,
        four_shots=render.max_sub_shots * render.sub_shot_seconds,
        spoken=spoken_seconds,
        low=spoken_seconds * rules.broll_ratio_min,
        high=spoken_seconds * rules.broll_ratio_max,
        rmin=rules.broll_ratio_min, rmax=rules.broll_ratio_max,
        intro=rules.intro_aroll_seconds,
        tags_min=rules.concept_tags_min, tags_max=rules.concept_tags_max,
    )


def script_block(script: Script) -> str:
    """O roteiro como texto, com os ids visiveis para o modelo referenciar."""
    partes = [
        f"Assunto: {script.subject}",
        f"Argumento: {script.argument}",
        f"O que o espectador leva: {script.viewer_takeaway}",
        "",
        "Roteiro:",
        "",
    ]
    for beat in script.beats:
        titulo = f" — {beat.title}" if beat.title else ""
        partes += [f"[beat {beat.id}{titulo}]", beat.text, ""]
    return "\n".join(partes)


def plan(
    script: Script,
    *,
    spoken_seconds: float,
    input_hash: str,
    config: AnthropicConfig,
    rules: EditorialConfig,
    render: RenderConfig,
    client: anthropic.Anthropic,
) -> Storyboard:
    """O roteiro vira beats visuais. Tenta de novo com o erro como feedback."""
    messages: list[dict] = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": script_block(script),
            "cache_control": {"type": "ephemeral"},
        }],
    }]

    problemas: list[str] = []

    for attempt in range(1, config.max_attempts + 1):
        log("storyboard.attempt", attempt=attempt, of=config.max_attempts,
            model=config.model)

        response = with_retries(
            lambda: client.with_options(timeout=config.timeout_seconds).messages.parse(
                model=config.model,
                max_tokens=config.max_tokens,
                system=[{
                    "type": "text",
                    "text": system_prompt(rules, render, spoken_seconds),
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=messages,
                output_format=PlannedStoryboard,
                output_config={"effort": config.effort},
            ),
            label=f"anthropic storyboard {config.model}",
        )

        if response.stop_reason == "refusal":
            raise StoryboardFailed(
                "a API recusou o storyboard "
                f"(categoria: {getattr(response.stop_details, 'category', None)})"
            )
        if response.stop_reason == "max_tokens":
            raise StoryboardFailed(
                f"o storyboard foi truncado em max_tokens={config.max_tokens}; "
                "aumente anthropic.max_tokens no config"
            )

        planned: PlannedStoryboard = response.parsed_output
        problemas = validate(planned.beats, script, rules, render, spoken_seconds)

        if not problemas:
            board = resolve(planned.beats, script, render, input_hash)
            log("storyboard.ok",
                imagens=board.n_images,
                planos=sum(b.sub_shots for b in board.beats),
                tela=f"{board.total_screen_seconds:.0f}s",
                cobertura=f"{board.total_screen_seconds / spoken_seconds:.0%}"
                          if spoken_seconds else "n/a",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens)
            return board

        log("storyboard.invalid", attempt=attempt, problemas=len(problemas))
        for problema in problemas:
            log("storyboard.problema", detail=problema)

        messages += [
            {"role": "assistant", "content": planned.model_dump_json()},
            {"role": "user", "content":
                "O storyboard nao passou na validacao:\n\n"
                + "\n".join(f"- {p}" for p in problemas)
                + "\n\nCorrija e devolva o storyboard inteiro."},
        ]

    raise StoryboardFailed(
        f"o storyboard nao ficou valido em {config.max_attempts} tentativas.\n"
        + "\n".join(f"- {p}" for p in problemas)
    )
