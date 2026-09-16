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
def example_config():
    """O config de exemplo como ele esta no repo, sem substituicao nenhuma."""
    return load_config(Path(__file__).resolve().parents[1] / "config.example.yaml")


@pytest.fixture(scope="session")
def config(example_config, rules):
    """O exemplo, com as regras editoriais trocadas pelas fixas do `rules`.

    Ler o resto do exemplo e proposital: estilo, orcamento, banco e render sao
    valores que a suite quer exercitar de verdade. Ja as regras editoriais sao
    calibracao pura, e herda-las fazia mudar o ritmo visual do exemplo quebrar
    testes que nao falam de ritmo. Ver o docstring de `rules`.
    """
    return example_config.model_copy(update={"editorial": rules})


@pytest.fixture(scope="session")
def rules():
    """Regras editoriais FIXAS, nao as do config.example.

    Herdar do exemplo acoplava a suite a calibracao: os spans destes testes
    caem no grid de 4.5s do fixture `transcript`, entao baixar
    `broll_max_seconds` no exemplo para dar mais dinamismo quebrava 22 testes
    que nao falam de calibracao nenhuma, so de regra.

    Sao os valores DO SPEC — b-roll de 8 a 25s, no maximo 3 trocas por
    minuto, 20s de intro em a-roll, 50 a 70% de cobertura. E contra eles que a
    suite foi escrita, e e o que os nomes dos testes dizem
    (`test_rejeita_broll_nos_primeiros_20s`). O exemplo foi derivando disso
    conforme a calibracao real, que e o trabalho dele; a suite ficar ancorada
    no spec e o que faz calibrar nao quebrar teste de regra.

    Que o exemplo em si tenha regras viaveis e assunto de
    `test_feasibility.test_config_de_exemplo_e_viavel`.
    """
    from pipeline.config import EditorialConfig

    return EditorialConfig(
        broll_min_seconds=8.0,
        broll_max_seconds=25.0,
        max_switches_per_minute=3,
        intro_aroll_seconds=20.0,
        broll_ratio_min=0.50,
        broll_ratio_max=0.70,
    )


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
