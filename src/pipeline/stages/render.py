"""Estagio 5: montagem em ffmpeg.

A ordem importa:

  1. pre-renderiza cada imagem na tela grande, uma vez (nao por frame);
  2. escreve o filtergraph em disco antes de executar, para depurar;
  3. renderiza a trilha de VIDEO, em um passe ou em chunks;
  4. muxa o video com o audio ORIGINAL por stream copy.

O audio so e tocado no passo 4, e la ele e copiado. E isso que preserva a
sincronia labial nos trechos de a-roll.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config, RenderConfig
from ..ffmpeg import probe
from ..ffmpeg import run as ffmpeg
from ..filtergraph import (
    Chunk, Overlay, audio_trim_chain, build_graph, build_inputs,
    fade_for, frame_align, plan_chunks, prep_size,
)
from ..log import log, stage
from ..schemas import EDL, SUB_SHOT_ORDER, Assets, Manifest, Region, Transcript, TrimPlan
from ..subtitles import write_ass
from ..util import text_hash


def upscale_factor(source_width: int, region: Region, cfg: RenderConfig) -> float:
    """Quanto a imagem e ampliada NA TELA num plano deste tipo.

    Nao e o fator do `prep`, que e igual para os dois: o plano cheio e
    reduzido de volta no fim do Ken Burns, enquanto o quadrante ja sai na
    resolucao de saida. Por isso o MESMO arquivo amplia o dobro num
    sub-plano — sao metade dos pixels da fonte preenchendo a mesma tela.

    Com uma imagem de 1344px de largura e saida em 1080p: 1.6x no plano
    cheio, 3.2x no quadrante. E o numero que separa as duas explicacoes para
    o "delirio" percebido — aderencia do modelo ao prompt, ou resolucao de
    origem insuficiente para o plano em que ela aparece.
    """
    fraction = 1.0 if region == "full" else 0.5
    return (cfg.width * cfg.ken_burns.zoom_max) / (source_width * fraction)


def sub_shots_enabled(cfg: RenderConfig) -> bool:
    return cfg.sub_shot_seconds > 0 and cfg.max_sub_shots > 1


def source_size(source: Path) -> tuple[int, int]:
    """(largura, altura) da imagem, ou (0, 0) se ela nao da para ler.

    (0, 0) e um resultado real e nao um caso hipotetico: download
    interrompido no meio deixa um arquivo que existe, tem bytes, e nao e
    imagem — e o `ffprobe` devolve `width: 0` em vez de falhar.
    """
    stream = next(
        (s for s in probe(source)["streams"] if s["codec_type"] == "video"), None
    )
    if stream is None:
        return 0, 0
    return int(stream.get("width") or 0), int(stream.get("height") or 0)


def _warn_if_small(relative: str, width: int, height: int, cfg: RenderConfig) -> None:
    """Avisa quando a fonte nao tem pixel para o plano em que ela vai entrar.

    Mede o pior plano que este config pode pedir, nao o melhor: com sub-plano
    ligado o quadrante e o que manda, porque e ele que vai preencher a tela
    com metade da imagem.
    """
    region: Region = "top_left" if sub_shots_enabled(cfg) else "full"
    factor = upscale_factor(width, region, cfg)
    if factor > cfg.upscale_warn_factor:
        log("render.warn",
            detail=f"{relative} tem {width}x{height} e amplia "
                   f"{factor:.1f}x no plano '{region}' (teto "
                   f"{cfg.upscale_warn_factor:g}x): gere a imagem maior, ou desligue "
                   f"o sub-plano com render.sub_shot_seconds: 0")


def prepare_images(assets: Assets, config: Config, work: Path) -> dict[int, Path]:
    """Escala cada imagem uma vez para a tela grande do Ken Burns.

    Fazer isso dentro do filtergraph custaria uma reescala da imagem inteira
    por frame — 450 reescalas identicas num segmento de 15s — e ainda inflaria
    o grafo. O resultado e cacheado por hash do arquivo de origem.
    """
    width, height = prep_size(config.render)
    prep_dir = work / "prep"
    prep_dir.mkdir(parents=True, exist_ok=True)

    prepared: dict[int, Path] = {}
    for item in assets.items:
        if item.path is None:
            continue
        # Beat que nao alinhou na fala sai de `assets.aligned.json` com
        # `segment_index = -1`. A imagem existe e foi paga, mas nao entra no
        # video: prepara-la seria uma reescala de 4304px jogada fora.
        if item.segment_index < 0:
            continue
        source = config.path(item.path)
        if not source.exists():
            log("render.warn", detail=f"imagem ausente, virou cor solida: {item.path}")
            continue

        # Arquivo ilegivel vale como arquivo ausente: vira cor solida, com o
        # nome no aviso para voce apagar e resolver de novo. Deixar seguir
        # matava o render inteiro no `ffmpeg` do prep, depois de as imagens ja
        # terem sido pagas — e por uma imagem so.
        origem_w, origem_h = source_size(source)
        if origem_w <= 0:
            log("render.warn",
                detail=f"imagem ilegivel, virou cor solida: {item.path} "
                       f"(apague o arquivo e rode o estagio de imagens de novo)")
            continue
        _warn_if_small(item.path, origem_w, origem_h, config.render)

        target = prep_dir / f"{text_hash(item.path, width, height)[:16]}.png"
        if not target.exists():
            ffmpeg([
                "-i", str(source),
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
                       f"crop={width}:{height}",
                "-frames:v", "1", str(target),
            ], label=f"preparar {source.name}")
        prepared[item.segment_index] = target

    log("render.prepared", images=len(prepared), size=f"{width}x{height}")
    return prepared


def sub_shot_count(duration: float, cfg: RenderConfig) -> int:
    """Quantos planos tirar de uma faixa de b-roll desta duracao.

    O numero sai da DURACAO da faixa e nao do `sub_shots` que o storyboard
    pediu, de proposito. O storyboard e escrito antes de existir gravacao, e
    o que ele escolhe de fato e o tempo de tela do beat
    (`sub_shots * seconds_per_shot`); onde aquele trecho caiu na fala real —
    e portanto o tamanho exato da faixa — so o alinhamento sabe, e ele cresce
    em segmento inteiro, entao a faixa costuma sobrar um pouco. Dividir a
    faixa real por `sub_shot_seconds` mantem o RITMO que o config pede; usar
    o numero do storyboard manteria a contagem e esticaria cada plano. O
    ritmo e o que se ve.

    Piso 1 e teto `max_sub_shots`: faixa curta nao vira meio plano, e faixa
    longa nao inventa regiao que nao existe. `sub_shot_seconds: 0` desliga.
    """
    if cfg.sub_shot_seconds <= 0:
        return 1
    return max(1, min(cfg.max_sub_shots, int(duration / cfg.sub_shot_seconds)))


def build_overlays(edl: EDL, prepared: dict[int, Path], config: Config) -> list[Overlay]:
    """Overlays de b-roll, com a faixa longa partida em sub-planos.

    Cada sub-plano recorta uma regiao diferente da MESMA imagem, e a emenda
    entre dois deles e corte seco: o fade de alpha existe para a fronteira
    com o a-roll, e no meio da faixa ele faria a sua imagem reaparecer por
    400ms. Por isso so o primeiro plano recebe fade de entrada e so o ultimo
    recebe fade de saida, os dois dimensionados pela faixa inteira.

    Sem imagem nao ha sub-plano: o fallback de cor solida nao tem regiao
    para recortar, e cortar entre dois pedacos da mesma cor nao existe.

    A direcao do Ken Burns anda por PLANO e nao por faixa, entao dois planos
    consecutivos da mesma imagem nunca se movem do mesmo jeito — o que, junto
    com a diagonal de `SUB_SHOT_ORDER`, e o que faz a emenda parecer corte.
    """
    directions = config.render.ken_burns.directions
    fps = config.render.fps
    overlays: list[Overlay] = []
    shot = 0

    for index, segment in enumerate(edl.segments):
        if segment.kind != "broll":
            continue
        image = prepared.get(index)
        start, end = frame_align(segment.start, fps), frame_align(segment.end, fps)
        count = 1 if image is None else sub_shot_count(end - start, config.render)
        fade = fade_for(end - start, config.render)
        edges = [frame_align(start + (end - start) * i / count, fps) for i in range(count)]
        edges.append(end)

        for position, region in enumerate(SUB_SHOT_ORDER[:count]):
            overlays.append(Overlay(
                start=edges[position], end=edges[position + 1],
                direction=directions[shot % len(directions)],
                image_path=image, region=region,
                fade_in=fade if position == 0 else 0.0,
                fade_out=fade if position == count - 1 else 0.0,
            ))
            shot += 1

    return overlays


def encode_args(config: Config) -> list[str]:
    return [
        "-c:v", "libx264", "-crf", str(config.render.crf),
        "-preset", config.render.preset, "-pix_fmt", "yuv420p",
        "-r", f"{config.render.fps:g}",
    ]


def render_chunk(
    chunk: Chunk, source: Path, transcript: Transcript, config: Config, work: Path
) -> Path:
    """Renderiza a trilha de video de um chunk. Sem audio: `-an`."""
    subtitle_path = None
    if config.subtitles.enabled:
        subtitle_path = write_ass(
            work / f"subs_{chunk.index:03d}.ass", transcript, config.subtitles,
            width=config.render.width, height=config.render.height,
            window=(chunk.start, chunk.end),
        )

    graph = build_graph(chunk, config.render, subtitle_path)
    graph_path = work / ("filtergraph.txt" if chunk.index == 0 else f"filtergraph_{chunk.index:03d}.txt")
    graph_path.write_text(graph, encoding="utf-8")

    target = work / "chunks" / f"chunk_{chunk.index:03d}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg([
        *build_inputs(chunk, source, config.render),
        "-filter_complex", graph.replace(";\n", ";"),
        "-map", "[vout]", "-an",
        *encode_args(config),
        str(target),
    ], label=f"chunk {chunk.index}")
    return target


def build_audio(
    plan: TrimPlan | None, source: Path, work: Path, config: Config
) -> tuple[Path, str]:
    """A trilha de audio da saida. Devolve (arquivo, codec para o mux).

    Sem corte, o audio nao e tocado: sai do arquivo original por stream copy
    no mux, byte a byte igual ao que entrou.

    Com corte, ele e cortado nos MESMOS instantes do video — o que preserva a
    sincronia labial — e encodado exatamente uma vez, num passe proprio que
    nao decodifica video. O invariante deixa de ser "nunca tocado" e passa a
    ser "cortado nos pontos escolhidos e nunca processado de outra forma":
    sem normalizacao, sem compressao, sem filtro.
    """
    if plan is None or not plan.enabled or len(plan.keep) <= 1:
        return source, "copy"

    target = work / "audio.m4a"
    keep = [(r.start, r.end) for r in plan.keep]
    ffmpeg([
        "-i", str(source), "-vn",
        "-filter_complex", audio_trim_chain(keep),
        "-map", "[aout]",
        "-c:a", "aac", "-b:a", str(config.render.audio_bitrate_kbps) + "k",
        str(target),
    ], label=f"cortar audio ({len(keep)} trechos)")
    log("render.audio", trechos=len(keep), bitrate=f"{config.render.audio_bitrate_kbps}k")
    return target, "copy"


def mux(video: Path, audio: Path, output: Path, codec: str) -> None:
    """Junta a trilha de video montada com a de audio."""
    ffmpeg([
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", codec,
        "-movflags", "+faststart", "-shortest",
        str(output),
    ], label="mux audio")


def run(
    manifest: Manifest, transcript: Transcript, edl: EDL, assets: Assets,
    config: Config, plan: TrimPlan | None = None,
) -> Path:
    work = config.work_dir / manifest.slug
    output = work / "final.mp4"
    source = Path(manifest.source_path)

    # `render.fps: 0` no config significa herdar o fps da entrada. Resolver
    # aqui, numa copia, faz todo o resto do estagio e do filtergraph ver um
    # numero concreto em vez de espalhar o `or manifest.fps`.
    config = config.model_copy(deep=True)
    if not config.render.fps:
        config.render.fps = manifest.fps
        log("render.fps", inherited=f"{manifest.fps:.3f}")
    elif abs(config.render.fps - manifest.fps) > 0.01:
        log("render.warn",
            detail=f"entrada a {manifest.fps:.2f}fps sai a {config.render.fps:g}fps; "
                   f"use render.fps: 0 para preservar")

    with stage("render", duration=f"{edl.duration:.1f}s", fps=f"{config.render.fps:g}"):
        prepared = prepare_images(assets, config, work)
        overlays = build_overlays(edl, prepared, config)
        duration = frame_align(edl.duration, config.render.fps)

        window = None
        if plan is not None and plan.enabled and len(plan.keep) > 1:
            def window(first: float, last: float):
                start = plan.original_time(first)
                return start, plan.original_time(last) - start, plan.keep_within(first, last)

        chunks = plan_chunks(overlays, duration, config.render, window)

        log("render.plan", chunks=len(chunks), planos=len(overlays),
            imagens=len(prepared),
            cortes=sum(len(c.keep) for c in chunks) if window else 0,
            mode="passe unico" if len(chunks) == 1 else "chunks + concat")

        rendered = [render_chunk(chunk, source, transcript, config, work) for chunk in chunks]

        if len(rendered) == 1:
            video = rendered[0]
        else:
            listing = work / "chunks" / "concat.txt"
            listing.write_text(
                "".join(f"file '{p.name}'\n" for p in rendered), encoding="utf-8"
            )
            video = work / "video.mp4"
            ffmpeg([
                "-f", "concat", "-safe", "0", "-i", str(listing),
                "-c", "copy", str(video),
            ], label="concat dos chunks")

        audio, audio_codec = build_audio(plan, source, work, config)
        mux(video, audio, output, audio_codec)
        log("render.ok", output=str(output))
        return output
