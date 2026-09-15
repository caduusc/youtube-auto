from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.config import load_config  # noqa: E402
from pipeline.schemas import PlannedEDL, PlannedSpan, Transcript, TranscriptSegment  # noqa: E402

SEG_SECONDS = 4.5
N_SEGMENTS = 200  # 200 * 4.5s = 900s = 15 min


@pytest.fixture(scope="session")
def config():
    return load_config(Path(__file__).resolve().parents[1] / "config.example.yaml")


@pytest.fixture(scope="session")
def rules(config):
    return config.editorial


@pytest.fixture
def transcript() -> Transcript:
    segments = [
        TranscriptSegment(
            id=i,
            start=round(i * SEG_SECONDS, 3),
            end=round((i + 1) * SEG_SECONDS, 3),
            text=f"frase numero {i} do transcript",
            words=[],
        )
        for i in range(N_SEGMENTS)
    ]
    return Transcript(
        input_hash="audiohash",
        language="pt",
        model="small",
        duration=round(N_SEGMENTS * SEG_SECONDS, 3),
        segments=segments,
    )


def span(kind: str, first: int, last: int) -> PlannedSpan:
    if kind == "broll":
        return PlannedSpan(
            kind="broll",
            first_segment=first,
            last_segment=last,
            concept="a wooden desk lit from the side with an open notebook",
            concept_tags=["desk", "notebook"],
        )
    return PlannedSpan(
        kind="aroll", first_segment=first, last_segment=last, concept="", concept_tags=[]
    )


def alternating(*, aroll_len: int, broll_len: int, n_segments: int = N_SEGMENTS) -> PlannedEDL:
    """Alterna a-roll e b-roll cobrindo o transcript inteiro.

    A primeira faixa de a-roll e estendida ate passar dos 20s de intro
    obrigatoria, senao a primeira troca cai dentro da janela protegida.
    """
    intro_len = max(aroll_len, math.ceil(20.0 / SEG_SECONDS) + 1)
    spans: list[PlannedSpan] = []
    cursor, kind = 0, "aroll"
    while cursor < n_segments:
        if kind == "broll":
            length = broll_len
        else:
            length = intro_len if not spans else aroll_len
        last = min(cursor + length - 1, n_segments - 1)
        spans.append(span(kind, cursor, last))
        cursor = last + 1
        kind = "broll" if kind == "aroll" else "aroll"
    return PlannedEDL(spans=spans)
