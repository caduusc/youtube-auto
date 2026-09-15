"""Estagio 1: valida o arquivo de entrada e extrai o audio."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..ffmpeg import parse_fps, probe
from ..ffmpeg import run as ffmpeg
from ..log import log, stage
from ..schemas import Manifest
from ..util import file_hash, now_iso, read_json_if_fresh, slugify, write_json


def slug_for(source: Path, content_hash: str) -> str:
    """Nome do arquivo mais um prefixo do hash do conteudo.

    O hash entra para que reexportar o mesmo arquivo com uma correcao nao
    reaproveite silenciosamente o trabalho feito em cima da versao antiga.
    """
    return f"{slugify(source.stem)}-{content_hash[:8]}"


def run(source: Path, config: Config) -> Manifest:
    source = Path(source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"arquivo de entrada nao encontrado: {source}")

    with stage("ingest", file=source.name):
        content_hash = file_hash(source)
        slug = slug_for(source, content_hash)
        work = config.work_dir / slug
        manifest_path = work / "manifest.json"

        cached = read_json_if_fresh(manifest_path, Manifest, content_hash)
        if cached is not None and (config.root / cached.audio_path).exists():
            log("ingest.cached", slug=slug)
            return cached

        info = probe(source)
        video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
        audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
        if video is None:
            raise RuntimeError(f"{source.name} nao tem trilha de video")
        if audio is None:
            raise RuntimeError(
                f"{source.name} nao tem trilha de audio — o pipeline monta imagem "
                "em cima da sua fala, sem audio nao ha o que montar"
            )

        duration = float(info["format"]["duration"])
        fps = parse_fps(video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1")
        width, height = int(video["width"]), int(video["height"])

        if duration < 60:
            log("ingest.warn", detail=f"video de {duration:.0f}s e curto para as regras editoriais")
        if height < config.render.height:
            log("ingest.warn",
                detail=f"entrada em {width}x{height}, abaixo da saida "
                       f"{config.render.width}x{config.render.height}: o a-roll vai ser ampliado")
        if fps <= 0:
            raise RuntimeError(f"ffprobe nao conseguiu determinar o fps de {source.name}")

        audio_path = work / "audio.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg(["-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(audio_path)], label="extrair audio")

        manifest = Manifest(
            input_hash=content_hash,
            slug=slug,
            source_path=str(source),
            duration=duration,
            width=width,
            height=height,
            fps=fps,
            video_codec=video["codec_name"],
            audio_codec=audio["codec_name"],
            audio_path=str(audio_path.relative_to(config.root)),
            audio_hash=file_hash(audio_path),
            created_at=now_iso(),
        )
        write_json(manifest_path, manifest)
        log("ingest.ok", slug=slug, duration=f"{duration:.1f}s",
            resolution=f"{width}x{height}", fps=f"{fps:.2f}")
        return manifest
