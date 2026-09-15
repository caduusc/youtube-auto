"""Geracao do arquivo ASS a partir do transcript.

Agrupa por palavra (usando os timestamps de palavra do faster-whisper) em
linhas de no maximo N caracteres, N linhas por evento. Se o transcript vier
sem palavras, cai para o texto do segmento inteiro.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import SubtitlesConfig
from .schemas import Transcript

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary},&H000000FF,{outline_color},&H00000000,{bold},0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


@dataclass
class Event:
    start: float
    end: float
    lines: list[str]


def ass_time(seconds: float) -> str:
    """H:MM:SS.cc — o ASS usa centesimos, nao milesimos."""
    seconds = max(0.0, seconds)
    hours, rest = divmod(seconds, 3600.0)
    minutes, secs = divmod(rest, 60.0)
    return f"{int(hours)}:{int(minutes):02d}:{int(secs):02d}.{int(round((secs - int(secs)) * 100)):02d}"


def group_events(transcript: Transcript, config: SubtitlesConfig) -> list[Event]:
    events: list[Event] = []

    for segment in transcript.segments:
        words = segment.words
        if not words:
            # sem timestamp por palavra: o segmento inteiro vira um evento
            events.extend(_wrap_plain(segment.text, segment.start, segment.end, config))
            continue

        lines: list[str] = []
        current = ""
        start = words[0].start
        last_end = words[0].end

        for word in words:
            token = word.word.strip()
            if not token:
                continue
            candidate = f"{current} {token}".strip()
            if current and len(candidate) > config.max_chars_per_line:
                lines.append(current)
                current = token
                if len(lines) == config.max_lines:
                    events.append(Event(start, last_end, lines))
                    lines, start = [], word.start
            else:
                current = candidate
            last_end = word.end

        if current:
            lines.append(current)
        if lines:
            events.append(Event(start, last_end, lines))

    return events


def _wrap_plain(text: str, start: float, end: float, config: SubtitlesConfig) -> list[Event]:
    words, lines, current = text.split(), [], ""
    for token in words:
        candidate = f"{current} {token}".strip()
        if current and len(candidate) > config.max_chars_per_line:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    if not lines:
        return []

    # divide o intervalo proporcionalmente entre os blocos de max_lines
    blocks = [lines[i : i + config.max_lines] for i in range(0, len(lines), config.max_lines)]
    span = (end - start) / len(blocks)
    return [Event(start + i * span, start + (i + 1) * span, block) for i, block in enumerate(blocks)]


def render_ass(
    transcript: Transcript,
    config: SubtitlesConfig,
    *,
    width: int,
    height: int,
    window: tuple[float, float] | None = None,
) -> str:
    """Monta o ASS. `window` recorta e desloca os tempos para o chunk.

    Sem o deslocamento, um chunk renderizado a partir de 120s mostraria a
    legenda do inicio do video.
    """
    body = HEADER.format(
        width=width, height=height,
        font_name=config.font_name, font_size=config.font_size,
        primary=config.primary_color, outline_color=config.outline_color,
        bold=-1 if config.bold else 0, outline=config.outline,
        shadow=config.shadow, alignment=config.alignment, margin_v=config.margin_v,
    )

    start_at, end_at = window if window else (0.0, float("inf"))
    rows: list[str] = []
    for event in group_events(transcript, config):
        if event.end <= start_at or event.start >= end_at:
            continue
        shifted_start = max(event.start, start_at) - start_at
        shifted_end = min(event.end, end_at) - start_at
        if shifted_end <= shifted_start:
            continue
        text = r"\N".join(line.replace("{", "(").replace("}", ")") for line in event.lines)
        rows.append(
            f"Dialogue: 0,{ass_time(shifted_start)},{ass_time(shifted_end)},Default,,0,0,0,,{text}"
        )

    return body + "\n".join(rows) + "\n"


def write_ass(path: Path, transcript: Transcript, config: SubtitlesConfig, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_ass(transcript, config, **kwargs), encoding="utf-8")
    return path
