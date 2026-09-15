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
from ..filtergraph import Chunk, Overlay, build_graph, build_inputs, frame_align, plan_chunks, prep_size
from ..log import log, stage
from ..schemas import EDL, Assets, Manifest, Transcript
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


def mux(video: Path, source: Path, output: Path, config: Config) -> None:
    """Junta a trilha de video montada com o audio original.

    `-c:a copy` e o ponto do pipeline inteiro: a trilha de audio que sai e
    byte a byte a que entrou.
    """
    ffmpeg([
        "-i", str(video), "-i", str(source),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "copy",
        "-movflags", "+faststart", "-shortest",
        str(output),
    ], label="mux audio original")


def run(
    manifest: Manifest, transcript: Transcript, edl: EDL, assets: Assets, config: Config
) -> Path:
    work = config.work_dir / manifest.slug
    output = work / "final.mp4"
    source = Path(manifest.source_path)

    with stage("render", duration=f"{edl.duration:.1f}s"):
        prepared = prepare_images(assets, config, work)
        overlays = build_overlays(edl, prepared, config)
        duration = frame_align(edl.duration, config.render.fps)
        chunks = plan_chunks(overlays, duration, config.render)

        log("render.plan", chunks=len(chunks), overlays=len(overlays),
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

        mux(video, source, output, config)
        log("render.ok", output=str(output))
        return output
