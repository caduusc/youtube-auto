"""Viabilidade das regras editoriais e a fronteira exata da proporcao.

Os casos aqui vieram de uma execucao real que falhou: um video de 133.9s com
17 segmentos, onde a EDL correta foi rejeitada com a mensagem "b-roll cobre
50% do video, abaixo do minimo de 50%".
"""

from __future__ import annotations

import pytest

from pipeline.edl import budget_for, validate
from pipeline.planner import PlanningFailed, first_request, plan
from pipeline.schemas import EDLSegment, Transcript, TranscriptSegment


def transcript_of(duration: float, n_segments: int) -> Transcript:
    step = duration / n_segments
    return Transcript(
        input_hash="h", language="pt", model="small", duration=duration,
        segments=[
            TranscriptSegment(id=i, start=round(i * step, 3),
                              end=round((i + 1) * step, 3), text=f"frase {i}", words=[])
            for i in range(n_segments)
        ],
    )


def segs(transcript: Transcript, layout: list[tuple[str, int, int]]) -> list[EDLSegment]:
    return [
        EDLSegment(kind=kind, first_segment=a, last_segment=b,
                   concept="a warm desk" if kind == "broll" else "",
                   concept_tags=["desk", "warm"] if kind == "broll" else [],
                   start=transcript.segments[a].start, end=transcript.segments[b].end)
        for kind, a, b in layout
    ]


# --------------------------------------------------------------------------
# a fronteira da proporcao
# --------------------------------------------------------------------------


def test_proporcao_exata_no_minimo_e_aceita(rules):
    """Uma EDL que acerta os 50% na mosca cai em 0.4999... por ponto
    flutuante. Comparar o ratio direto a rejeitava com uma mensagem que o
    modelo nao tem como corrigir, porque nada estava errado."""
    t = transcript_of(120.0, 12)   # 12 segmentos de 10s
    # aroll 0-2 (30s) | broll 3-4 (20s) | aroll 5 (10s) | broll 6-7 (20s)
    # aroll 8 (10s) | broll 9-10 (20s) | aroll 11 (10s)  -> broll = 60s = 50%
    layout = [("aroll", 0, 2), ("broll", 3, 4), ("aroll", 5, 5), ("broll", 6, 7),
              ("aroll", 8, 8), ("broll", 9, 10), ("aroll", 11, 11)]
    segments = segs(t, layout)
    broll = sum(s.duration for s in segments if s.kind == "broll")
    assert broll == pytest.approx(60.0), broll

    errors = validate(segments, t, rules)
    assert not any("abaixo do minimo" in e for e in errors), errors


def test_teto_nominal_de_70_por_cento_e_inalcancavel(rules):
    """Com faixas de L segundos e proporcao r, a taxa de trocas e 120*r/L —
    nao depende da duracao. Com faixa de no maximo 25s e 3 trocas/min, a
    proporcao para de subir em 62.5%: o teto nominal de 70% do config nunca
    pode ser atingido, e um modelo que mire no meio de 50-70% viola o ritmo
    em toda tentativa. O orcamento precisa reportar a janela REAL."""
    assert rules.broll_ratio_max == 0.70
    alcancavel = rules.broll_max_seconds * rules.max_switches_per_minute / 120
    assert alcancavel < rules.broll_ratio_max
    assert alcancavel == pytest.approx(0.625)

    for duration in (133.9, 300.0, 900.0, 1800.0):
        b = budget_for(duration, rules)
        assert b.ratio_max <= rules.broll_ratio_max
        # e o teto reportado cabe nas faixas que o ritmo permite
        assert b.broll_seconds_max <= b.n_broll_max * rules.broll_max_seconds + 1e-6


def test_prompt_avisa_do_teto_real(rules):
    text = budget_for(900.0, rules).as_prompt()
    assert "para baixo" in text
    assert "independente do teto" in text
    # e manda mirar no meio, nao no topo
    assert "MEIO" in text


def test_proporcao_de_fato_baixa_ainda_e_rejeitada(rules):
    t = transcript_of(120.0, 12)
    layout = [("aroll", 0, 8), ("broll", 9, 10), ("aroll", 11, 11)]
    errors = validate(segs(t, layout), t, rules)
    assert any("abaixo do minimo" in e for e in errors), errors
    # e a mensagem diz quantos segundos faltam, nao so a porcentagem
    assert any("faltam" in e and "s de b-roll" in e for e in errors), errors


# --------------------------------------------------------------------------
# orcamento derivado
# --------------------------------------------------------------------------


def test_video_do_usuario_e_viavel_com_3_brolls(rules):
    """133.9s: o espaco de solucao e um unico ponto."""
    b = budget_for(133.9, rules)
    assert b.feasible
    assert b.n_broll_min == b.n_broll_max == 3
    assert b.max_switches == 6


def test_video_de_um_minuto_e_inviavel(rules):
    """Com 20s de intro obrigatoria, 70% de b-roll nao deixa a-roll suficiente."""
    b = budget_for(60.0, rules)
    assert not b.feasible
    assert "intro" in b.reason


def test_video_longo_tem_folga(rules):
    b = budget_for(900.0, rules)
    assert b.feasible
    assert b.n_broll_max > b.n_broll_min   # varias configuracoes servem


def test_orcamento_vira_numero_no_prompt(rules):
    t = transcript_of(133.9, 17)
    text = first_request(t, budget_for(133.9, rules))
    assert "no maximo 6 trocas de tela no total" in text
    assert "exatamente 3 faixas de b-roll" in text
    # o teto e 75.0s (3 faixas x 25s), nao os 93.7s do 70% nominal
    assert "67.0s e 75.0s" in text
    assert "~7.9s cada em media" in text


# --------------------------------------------------------------------------
# falha rapida
# --------------------------------------------------------------------------


def test_video_inviavel_nao_chama_a_api(config):
    """Descobrir a inviabilidade custando 3 tentativas de Opus e caro."""

    class ExplodingClient:
        def with_options(self, **_):
            raise AssertionError("a API foi chamada para um video inviavel")

        messages = None

    with pytest.raises(PlanningFailed, match="nao tem solucao para este video"):
        plan(transcript_of(60.0, 8), config=config.anthropic,
             rules=config.editorial, client=ExplodingClient())


def test_mensagem_de_inviabilidade_aponta_os_knobs(config):
    with pytest.raises(PlanningFailed) as excinfo:
        plan(transcript_of(60.0, 8), config=config.anthropic,
             rules=config.editorial, client=object())
    message = str(excinfo.value)
    assert "Nenhuma chamada a API foi feita" in message
    assert "intro_aroll_seconds" in message
    assert "broll_ratio_min" in message


# --------------------------------------------------------------------------
# a regressao completa: o video que falhou de verdade
# --------------------------------------------------------------------------


def test_edl_valida_existe_para_o_video_que_falhou(rules):
    """133.9s, 17 segmentos de ~7.9s. Na execucao real, as 3 tentativas
    falharam: duas por ritmo (3.14 trocas/min) e uma pela fronteira de 50%
    que o bug de ponto flutuante rejeitava. Este teste fixa que uma EDL
    valida existe e passa — o espaco de solucao e um unico ponto."""
    t = transcript_of(133.9, 17)
    layout = [
        ("aroll", 0, 2),    # 23.6s, cobre a intro de 20s
        ("broll", 3, 5),    # 23.6s
        ("aroll", 6, 6),    # 7.9s
        ("broll", 7, 9),    # 23.6s
        ("aroll", 10, 10),  # 7.9s
        ("broll", 11, 13),  # 23.6s
        ("aroll", 14, 16),  # 23.6s
    ]
    segments = segs(t, layout)

    assert validate(segments, t, rules) == []

    broll = sum(s.duration for s in segments if s.kind == "broll")
    b = budget_for(133.9, rules)
    assert len(segments) == b.max_spans
    assert sum(1 for s in segments if s.kind == "broll") == b.n_broll_max
    assert b.broll_seconds_min - 0.05 <= broll <= b.broll_seconds_max + 0.05


def test_uma_troca_a_mais_ainda_e_rejeitada(rules):
    """O que o modelo entregou nas tentativas 1 e 3: 8 faixas, 7 trocas."""
    t = transcript_of(133.9, 17)
    layout = [
        ("aroll", 0, 2), ("broll", 3, 5), ("aroll", 6, 6), ("broll", 7, 9),
        ("aroll", 10, 10), ("broll", 11, 12), ("aroll", 13, 13), ("broll", 14, 16),
    ]
    errors = validate(segs(t, layout), t, rules)
    assert any("por minuto no video inteiro" in e for e in errors), errors
