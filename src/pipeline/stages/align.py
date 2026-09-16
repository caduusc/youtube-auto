"""Estagio 6: onde cada imagem aprovada cai na fala que voce gravou.

E o estagio que faz o caminho roteiro-primeiro fechar. Ele le tres coisas que
existiam ANTES da gravacao — o storyboard aprovado, as imagens aprovadas — e
uma que existe depois — o transcript — e escreve as duas pecas que o render
precisa: a EDL e um `assets` indexado por segmento.

Ele trabalha no transcript CORTADO (`transcript.trimmed.json`), e nao no
original, pela mesma razao que o `plan` do outro caminho: e essa a timeline em
que o video sai. Alinhar no transcript original poria cada imagem alguns
decimos de segundo adiante do trecho que ela ilustra, proporcionalmente ao
corte acumulado antes dela — um erro que cresce ao longo do video e que
ninguem veria como erro de alinhamento.

Tres artefatos, e cada um existe por um motivo diferente:

    edl.json               o que o render consome
    assets.aligned.json    as imagens com `segment_index` preenchido
    align.json             ONDE cada beat casou, e como — para voce revisar

O terceiro nao alimenta nada: ele e o relatorio. Um beat que casou por
similaridade fraca, ou que nao casou, e a unica coisa nesta etapa que pede o
seu olho, e sem gravar isso em disco a informacao morreria no log.

Este estagio nao valida a EDL contra `editorial.broll_min_seconds` /
`broll_max_seconds`, e a omissao e deliberada: no caminho roteiro-primeiro a
duracao de cada b-roll e decidida por `sub_shots * sub_shot_seconds`, que voce
aprovou beat por beat. Reportar que um beat de 3 sub-planos (7.5s) esta abaixo
de um minimo de 8s seria re-litigar uma decisao ja tomada, e ensinaria voce a
ignorar os avisos. O que a gravacao PODE mudar — a cobertura e a taxa de troca,
que dependem de quao rapido voce falou — sai como aviso de verdade.
"""

from __future__ import annotations

from .. import approval
from ..align import aligned_assets, edl_from_spans, place, spans_for
from ..config import Config
from ..embed import SentenceTransformerEmbedder, resolve_model
from ..log import log, stage
from ..schemas import (
    EDL, Assets, BeatSpan, Placements, Script, Storyboard, Transcript,
)
from ..util import read_json_if_fresh, text_hash, write_json


def cache_key(board: Storyboard, transcript: Transcript, assets: Assets,
              config: Config) -> str:
    """Tudo que muda ONDE as imagens caem.

    `min_similarity` entra porque mexe na fronteira entre casar e ficar orfao;
    `intro_aroll_seconds` porque e o piso que empurra faixa para frente; o
    hash dos assets porque um beat que perdeu a imagem nao deve continuar
    ocupando tela.
    """
    return text_hash(
        board.input_hash,
        transcript.digest(),
        assets.input_hash,
        config.align.min_similarity,
        config.editorial.intro_aroll_seconds,
    )


def warnings_for(edl: EDL, config: Config) -> list[str]:
    """O que a gravacao pode ter mudado em relacao ao storyboard aprovado.

    O storyboard estimou a cobertura a partir de `words_per_minute`. Se voce
    falou mais rapido que a estimativa, a mesma cobertura em segundos vira uma
    fracao maior do video; mais devagar, menor. Nao e erro de ninguem — e
    informacao que so a gravacao revela.
    """
    rules = config.editorial
    avisos: list[str] = []
    ratio = edl.stats.broll_ratio

    if ratio < rules.broll_ratio_min:
        avisos.append(
            f"b-roll cobre {ratio:.0%} do video, abaixo da faixa de "
            f"{rules.broll_ratio_min:.0%}-{rules.broll_ratio_max:.0%}: voce falou "
            f"mais que o roteiro previa, ou beats ficaram orfaos"
        )
    elif ratio > rules.broll_ratio_max:
        avisos.append(
            f"b-roll cobre {ratio:.0%} do video, acima da faixa de "
            f"{rules.broll_ratio_min:.0%}-{rules.broll_ratio_max:.0%}: voce falou "
            f"menos que o roteiro previa e a sua imagem aparece pouco"
        )

    if edl.stats.switches_per_minute_max > rules.max_switches_per_minute:
        avisos.append(
            f"pico de {edl.stats.switches_per_minute_max:.0f} trocas de tela numa "
            f"janela de 60s, acima de {rules.max_switches_per_minute}: os beats "
            f"cairam agrupados na fala"
        )

    return avisos


def run(
    script: Script,
    board: Storyboard,
    transcript: Transcript,
    assets: Assets,
    config: Config,
) -> tuple[EDL, Assets]:
    work = config.work_dir / script.slug
    edl_path = work / "edl.json"
    aligned_path = work / "assets.aligned.json"
    key = cache_key(board, transcript, assets, config)

    # O portao das imagens: sao elas que custaram dinheiro, e este e o primeiro
    # estagio depois da gravacao — falhar aqui e mais barato que falhar depois
    # de uma hora de encode.
    # Pelo DIGEST e nao pelo `input_hash`: regerar uma imagem nao muda de
    # onde o `assets` foi derivado, so o que ele contem. Ver `Assets.digest`.
    approval.require(
        work, "images", assets.digest(),
        what="As imagens",
        command=f"pipeline approve images {script.slug}",
    )

    # Os TRES artefatos entram na conferencia. Sem o relatorio aqui, um
    # `align.json` velho (ou apagado) sobrevivia a um cache hit, e o
    # `pipeline shoot` imprimiria o alinhamento de outra rodada — ou morreria
    # lendo um arquivo que nao existe.
    cached_edl = read_json_if_fresh(edl_path, EDL, key)
    cached_assets = read_json_if_fresh(aligned_path, Assets, key)
    cached_report = read_json_if_fresh(work / "align.json", Placements, key)
    if None not in (cached_edl, cached_assets, cached_report):
        log("align.cached", slug=script.slug, broll=cached_edl.stats.n_broll)
        return cached_edl, cached_assets

    with stage("align", slug=script.slug, beats=board.n_images,
               segmentos=len(transcript.segments)):
        # O embedder e construido sempre e carregado nunca, a nao ser que um
        # ancora falhe na busca literal: `SentenceTransformerEmbedder` so
        # importa o torch no primeiro `embed()`. Ver `embed.py`.
        identity, load_path = resolve_model(config.bank.embedding_model, config.root)
        embedder = SentenceTransformerEmbedder(identity, load_path)

        placements = place(board, transcript, rules=config.align, embedder=embedder)
        spans = spans_for(placements, board, transcript,
                          intro_seconds=config.editorial.intro_aroll_seconds)
        edl = edl_from_spans(spans, board, transcript, input_hash=key)
        avisos = warnings_for(edl, config)
        for aviso in avisos:
            log("align.warn", detail=aviso)

        orfaos = [c for c in placements if c.segment < 0]
        if orfaos:
            log("align.orphans", beats=",".join(str(c.beat_id) for c in orfaos),
                detail="imagens pagas que nao entram no video; veja align.json")

        aligned = aligned_assets(assets, spans, edl.segments, input_hash=key)

        write_json(edl_path, edl)
        write_json(aligned_path, aligned)
        write_json(work / "align.json", Placements(
            input_hash=key, placements=placements,
            spans=[
                BeatSpan(beat_id=beat_id, first_segment=first, last_segment=last,
                         start=transcript.segments[first].start,
                         end=transcript.segments[last].end)
                for beat_id, first, last in spans
            ],
            n_orphans=len(orfaos), warnings=avisos,
        ))

        log("align.ready", broll=edl.stats.n_broll,
            cobertura=f"{edl.stats.broll_ratio:.0%}",
            orfaos=len(orfaos))
        return edl, aligned


def render_text(placements: Placements, board: Storyboard) -> str:
    """O alinhamento para revisar no terminal.

    Mostra o ancora E a faixa, que sao coisas diferentes: o piso do intro e o
    cursor de nao-sobreposicao empurram a imagem para frente sem mexer no
    ancora. So o segmento do ancora fazia o relatorio dizer "segmento 0" para
    uma imagem que entra aos 10s.

    `similarity` fica visivel porque e ele que separa "voce disse o trecho
    como estava escrito" de "o embedding achou um lugar parecido": o segundo
    caso merece o seu olho antes de renderizar.
    """
    por_id = {b.beat_id: b for b in board.beats}
    faixas = {f.beat_id: f for f in placements.spans}
    linhas = ["", "  Alinhamento", "  " + "-" * 72]
    for colocado in placements.placements:
        beat = por_id.get(colocado.beat_id)
        faixa = faixas.get(colocado.beat_id)
        onde = (f"{faixa.start:7.1f}s -{faixa.end:7.1f}s" if faixa is not None
                else "        ORFAO       ")
        cos = "" if colocado.method == "literal" else f"  cos {colocado.similarity:.2f}"
        linhas.append(f"  beat {colocado.beat_id:<3} {onde}  {colocado.method}{cos}")
        if beat is not None:
            linhas.append(f'    ancora: "{beat.script_anchor}"'
                          f"  (achado no segmento {colocado.segment})")
    linhas.append("  " + "-" * 72)
    if placements.n_orphans:
        linhas.append(f"  {placements.n_orphans} imagem(ns) sem lugar na fala — "
                      f"foram pagas e nao entram")
    for aviso in placements.warnings:
        linhas.append(f"  ! {aviso}")
    linhas.append("")
    return "\n".join(linhas)
