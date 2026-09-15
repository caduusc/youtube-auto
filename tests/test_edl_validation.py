"""O validador da EDL. Cada regra editorial tem um teste que a viola."""

from __future__ import annotations

import pytest
from conftest import SEG_SECONDS, alternating, span

from pipeline.edl import build, resolve, stats, validate
from pipeline.schemas import PlannedEDL


def check(planned, transcript, rules):
    return validate(resolve(planned, transcript), transcript, rules)


# --------------------------------------------------------------------------
# o caso bom
# --------------------------------------------------------------------------


def test_edl_valida_passa(transcript, rules):
    # a=4 (18s) + b=5 (22.5s): ciclo de 40.5s, 2.96 trocas/min, 56% de b-roll.
    # E a unica familia de combinacoes que satisfaz as tres regras de uma vez.
    assert check(alternating(aroll_len=4, broll_len=5), transcript, rules) == []


def test_stats_reflete_a_edl(transcript, rules):
    result = stats(resolve(alternating(aroll_len=4, broll_len=5), transcript), transcript.duration)
    assert 0.50 <= result.broll_ratio <= 0.70
    assert result.n_broll > 0 and result.n_aroll > 0
    assert result.switches_per_minute <= rules.max_switches_per_minute
    # a janela deslizante pega uma troca a mais que a taxa media, por borda
    assert result.switches_per_minute_max <= rules.max_switches_per_minute + 1


# --------------------------------------------------------------------------
# uma regra por teste
# --------------------------------------------------------------------------


def test_rejeita_broll_nos_primeiros_20s(transcript, rules):
    planned = PlannedEDL(spans=[span("broll", 0, 3), span("aroll", 4, 199)])
    errors = check(planned, transcript, rules)
    assert any("primeiros 20s" in e for e in errors), errors


def test_rejeita_broll_curto_demais(transcript, rules):
    # 1 segmento = 4.5s, abaixo do minimo de 8s
    planned = PlannedEDL(spans=[span("aroll", 0, 5), span("broll", 6, 6), span("aroll", 7, 199)])
    errors = check(planned, transcript, rules)
    assert any("abaixo do" in e and "minimo de 8s" in e for e in errors), errors


def test_rejeita_broll_longo_demais(transcript, rules):
    # 7 segmentos = 31.5s, acima do maximo de 25s
    planned = PlannedEDL(spans=[span("aroll", 0, 5), span("broll", 6, 12), span("aroll", 13, 199)])
    errors = check(planned, transcript, rules)
    assert any("acima do" in e and "maximo de 25s" in e for e in errors), errors


def test_rejeita_ritmo_acelerado(transcript, rules):
    """b-roll de 9s picado com a-roll de 4.5s: ~8.9 trocas/min."""
    spans, cursor = [span("aroll", 0, 5)], 6
    kind = "broll"
    while cursor < 200:
        length = 2 if kind == "broll" else 1
        last = min(cursor + length - 1, 199)
        spans.append(span(kind, cursor, last))
        cursor = last + 1
        kind = "aroll" if kind == "broll" else "broll"
    errors = check(PlannedEDL(spans=spans), transcript, rules)
    assert any("por minuto no video inteiro" in e for e in errors), errors
    assert any("em qualquer janela de 60s" in e for e in errors), errors


def test_rejeita_pouco_broll(transcript, rules):
    planned = PlannedEDL(spans=[span("aroll", 0, 180), span("broll", 181, 184), span("aroll", 185, 199)])
    errors = check(planned, transcript, rules)
    assert any("abaixo do minimo" in e and "faltam" in e for e in errors), errors


def test_rejeita_broll_demais(transcript, rules):
    planned = PlannedEDL(spans=[span("aroll", 0, 5), span("broll", 6, 10), span("aroll", 11, 12),
                                span("broll", 13, 17), span("aroll", 18, 19), span("broll", 20, 24),
                                span("aroll", 25, 25), span("broll", 26, 30), span("aroll", 31, 31),
                                span("broll", 32, 36), span("aroll", 37, 37), span("broll", 38, 199)])
    errors = check(planned, transcript, rules)
    assert any("acima do maximo" in e and "devolva" in e for e in errors), errors


def test_rejeita_buraco_na_timeline(transcript, rules):
    planned = PlannedEDL(spans=[span("aroll", 0, 5), span("broll", 10, 13), span("aroll", 14, 199)])
    errors = check(planned, transcript, rules)
    assert any("nao encaixam" in e for e in errors), errors


def test_rejeita_timeline_incompleta(transcript, rules):
    planned = PlannedEDL(spans=[span("aroll", 0, 5), span("broll", 6, 9)])
    errors = check(planned, transcript, rules)
    assert any("cobrir o video inteiro" in e for e in errors), errors


def test_rejeita_faixas_do_mesmo_tipo_coladas(transcript, rules):
    planned = PlannedEDL(spans=[span("aroll", 0, 5), span("aroll", 6, 9), span("broll", 10, 199)])
    errors = check(planned, transcript, rules)
    assert any("consecutivas" in e for e in errors), errors


def test_rejeita_broll_sem_concept(transcript, rules):
    bad = span("broll", 6, 9)
    bad.concept = "   "
    planned = PlannedEDL(spans=[span("aroll", 0, 5), bad, span("aroll", 10, 199)])
    errors = check(planned, transcript, rules)
    assert any("sem `concept`" in e for e in errors), errors


@pytest.mark.parametrize("tags", [[], ["one"], ["a", "b", "c", "d", "e"]])
def test_rejeita_contagem_de_tags_fora_da_faixa(transcript, rules, tags):
    bad = span("broll", 6, 9)
    bad.concept_tags = tags
    planned = PlannedEDL(spans=[span("aroll", 0, 5), bad, span("aroll", 10, 199)])
    errors = check(planned, transcript, rules)
    assert any("concept_tags" in e for e in errors), errors


def test_rejeita_aroll_com_concept(transcript, rules):
    bad = span("aroll", 6, 9)
    bad.concept = "should not be here"
    planned = PlannedEDL(spans=[span("broll", 0, 5), bad, span("aroll", 10, 199)])
    errors = check(planned, transcript, rules)
    assert any("so b-roll tem esses campos" in e for e in errors), errors


# --------------------------------------------------------------------------
# fronteira de frase: garantida por construcao, nao por regra
# --------------------------------------------------------------------------


def test_fronteiras_caem_sempre_em_limite_de_segmento(transcript, rules):
    """Como o modelo devolve indices, nao tempos, cortar no meio de uma
    frase nao e representavel. Este teste fixa essa propriedade."""
    planned = alternating(aroll_len=4, broll_len=5)
    boundaries = {round(s.start, 3) for s in transcript.segments}
    boundaries |= {round(s.end, 3) for s in transcript.segments}
    for segment in resolve(planned, transcript):
        assert round(segment.start, 3) in boundaries
        assert round(segment.end, 3) in boundaries


def test_indice_fora_do_transcript_levanta(transcript):
    with pytest.raises(IndexError, match="mas o transcript tem indices"):
        resolve(PlannedEDL(spans=[span("aroll", 0, 999)]), transcript)


def test_faixa_invertida_levanta(transcript):
    with pytest.raises(IndexError, match="antes de first_segment"):
        resolve(PlannedEDL(spans=[span("aroll", 10, 4)]), transcript)


def test_build_carrega_hash_do_transcript(transcript):
    planned = alternating(aroll_len=4, broll_len=5)
    result = build(planned, transcript, model="claude-opus-5", attempts=2)
    assert result.input_hash == transcript.digest()
    assert result.attempts == 2
    assert len(result.broll) == result.stats.n_broll
