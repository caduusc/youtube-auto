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

from ..config import Config
from ..ffmpeg import run as ffmpeg
from ..filtergraph import (
    Chunk, Overlay, audio_trim_chain, build_graph, build_inputs,
    frame_align, plan_chunks, prep_size,
)
from ..log import log, stage
from ..schemas import EDL, Assets, Manifest, Transcript, TrimPlan
from ..subtitles import write_ass
from ..util import text_hash


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
        source = config.path(item.path)
        if not source.exists():
            log("render.warn", detail=f"imagem ausente, virou cor solida: {item.path}")
            continue

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


def build_overlays(edl: EDL, prepared: dict[int, Path], config: Config) -> list[Overlay]:
    """Um overlay por segmento de b-roll, com a direcao do Ken Burns
    alternando entre segmentos consecutivos."""
    directions = config.render.ken_burns.directions
    fps = config.render.fps
    overlays: list[Overlay] = []
    ordinal = 0

    for index, segment in enumerate(edl.segments):
        if segment.kind != "broll":
            continue
        overlays.append(Overlay(
            start=frame_align(segment.start, fps),
            end=frame_align(segment.end, fps),
            direction=directions[ordinal % len(directions)],
            image_path=prepared.get(index),
        ))
        ordinal += 1

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

        log("render.plan", chunks=len(chunks), overlays=len(overlays),
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
