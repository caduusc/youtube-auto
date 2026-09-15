"""ffmpeg e ffprobe via subprocess. Sem wrappers."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .log import log


def _binary(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} nao encontrado no PATH — instale o ffmpeg")
    return found


def probe(path: Path) -> dict:
    out = subprocess.run(
        [_binary("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def run(args: list[str], *, label: str) -> None:
    """Roda ffmpeg. Em caso de erro, mostra o stderr — e onde o ffmpeg fala."""
    cmd = [_binary("ffmpeg"), "-hide_banner", "-nostdin", "-y", *args]
    log("ffmpeg.run", label=label, args=len(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-25:])
        raise RuntimeError(f"ffmpeg falhou em {label} (rc={result.returncode}):\n{tail}")


def parse_fps(rate: str) -> float:
    """'30000/1001' -> 29.97"""
    if "/" in rate:
        num, den = rate.split("/", 1)
        return float(num) / float(den) if float(den) else 0.0
    return float(rate)
