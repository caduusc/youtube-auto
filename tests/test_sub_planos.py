"""Sub-planos: varios planos tirados da mesma imagem.

O que esta sob teste aqui e a aritmetica — quantos planos, que regiao, que
fade em que ponta. A prova de que os recortes mostram de fato regioes
DIFERENTES da imagem esta em `test_render_smoke.py`, com ffmpeg de verdade.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pipeline.config import RenderConfig
from pipeline.schemas import EDL, EDLSegment, EDLStats
from pipeline.stages.render import (
    build_overlays, sub_shot_count, sub_shots_enabled, upscale_factor,
)

IMAGEM = Path("/prep/a1b2c3.png")


def edl_com(*faixas: tuple[str, float, float]) -> EDL:
    """EDL minima a partir de (kind, start, end)."""
    segments = [
        EDLSegment(kind=kind, first_segment=i, last_segment=i, start=start, end=end)
        for i, (kind, start, end) in enumerate(faixas)
    ]
    broll = sum(s.end - s.start for s in segments if s.kind == "broll")
    return EDL(
        input_hash="x", model="test", attempts=1,
        duration=max(s.end for s in segments), segments=segments,
        stats=EDLStats(broll_ratio=0.5, switches_per_minute=1.0,
                       switches_per_minute_max=2.0, n_aroll=0, n_broll=0,
                       broll_seconds=broll),
    )


def prepared(*indices: int) -> dict[int, Path]:
    return {i: IMAGEM for i in indices}


# --------------------------------------------------------------------------
# quantos planos
# --------------------------------------------------------------------------


@pytest.mark.parametrize("duracao,esperado", [
    (2.4, 1),      # abaixo de um plano: nao vira meio plano
    (2.5, 1),
    (5.0, 2),
    (9.0, 3),
    (10.0, 4),
    (25.0, 4),     # teto de max_sub_shots, nao de duracao
])
def test_contagem_sai_da_duracao(duracao, esperado):
    assert sub_shot_count(duracao, RenderConfig()) == esperado


def test_sub_shot_seconds_zero_desliga():
    cfg = RenderConfig(sub_shot_seconds=0.0)
    assert not sub_shots_enabled(cfg)
    assert sub_shot_count(20.0, cfg) == 1


def test_max_sub_shots_um_desliga():
    assert not sub_shots_enabled(RenderConfig(max_sub_shots=1))


# --------------------------------------------------------------------------
# a expansao em overlays
# --------------------------------------------------------------------------


def test_faixa_longa_vira_planos_de_regioes_diferentes(config):
    planos = build_overlays(edl_com(("broll", 0.0, 10.0)), prepared(0), config)

    assert [p.region for p in planos] == ["full", "top_left", "bottom_right", "top_right"]
    assert len({p.region for p in planos}) == 4, "duas regioes iguais na mesma faixa"
    assert all(p.image_path == IMAGEM for p in planos), "sub-plano pagou imagem nova"


def test_os_planos_ladrilham_a_faixa_sem_buraco(config):
    planos = build_overlays(edl_com(("broll", 12.0, 22.0)), prepared(0), config)

    assert planos[0].start == pytest.approx(12.0)
    assert planos[-1].end == pytest.approx(22.0)
    for anterior, seguinte in zip(planos, planos[1:]):
        assert seguinte.start == anterior.end, "buraco ou sobreposicao entre planos"


def test_so_as_pontas_da_faixa_tem_fade(config):
    """Um fade no meio faria a imagem do usuario reaparecer entre dois planos
    da mesma foto. O corte ali e seco."""
    planos = build_overlays(edl_com(("broll", 0.0, 10.0)), prepared(0), config)

    assert planos[0].fade_in > 0 and planos[0].fade_out == 0.0
    assert [p.fade_in for p in planos[1:-1]] == [0.0, 0.0]
    assert [p.fade_out for p in planos[1:-1]] == [0.0, 0.0]
    assert planos[-1].fade_in == 0.0 and planos[-1].fade_out > 0


def test_fade_e_dimensionado_pela_faixa_e_nao_pelo_plano(config):
    """O que o espectador ve entrar e a faixa inteira: um plano de 2.5s dentro
    de uma faixa de 10s nao deve receber o fade encurtado de 2.5s."""
    planos = build_overlays(edl_com(("broll", 0.0, 10.0)), prepared(0), config)
    assert planos[0].fade_in == config.render.crossfade_seconds


def test_direcao_anda_por_plano_e_nao_por_faixa(config):
    """Dois planos consecutivos da mesma imagem com o mesmo movimento
    pareceriam um salto dentro da mesma moldura em vez de um corte."""
    planos = build_overlays(edl_com(("broll", 0.0, 10.0)), prepared(0), config)
    direcoes = [p.direction for p in planos]
    assert direcoes == config.render.ken_burns.directions[:4]
    assert len(set(direcoes)) == 4


def test_a_direcao_continua_andando_entre_faixas(config):
    """O contador e global: a segunda faixa nao recomeca no mesmo movimento
    com que a primeira comecou."""
    planos = build_overlays(
        edl_com(("broll", 0.0, 5.0), ("aroll", 5.0, 9.0), ("broll", 9.0, 14.0)),
        prepared(0, 2), config)
    assert [p.direction for p in planos] == config.render.ken_burns.directions[:4]


def test_faixa_sem_imagem_nao_vira_sub_plano(config):
    """Cor solida nao tem regiao para recortar, e cortar entre dois pedacos da
    mesma cor nao existe."""
    planos = build_overlays(edl_com(("broll", 0.0, 20.0)), {}, config)

    assert len(planos) == 1
    assert planos[0].image_path is None
    assert planos[0].region == "full"


def test_aroll_nao_entra(config):
    planos = build_overlays(
        edl_com(("aroll", 0.0, 20.0), ("broll", 20.0, 25.0)), prepared(1), config)
    assert len(planos) == 2
    assert planos[0].start == pytest.approx(20.0)


def test_desligado_volta_a_um_plano_por_faixa(config):
    cfg = config.model_copy(deep=True)
    cfg.render.sub_shot_seconds = 0.0
    planos = build_overlays(edl_com(("broll", 0.0, 20.0)), prepared(0), cfg)

    assert len(planos) == 1
    assert planos[0].region == "full"
    assert planos[0].fade_in > 0 and planos[0].fade_out > 0


# --------------------------------------------------------------------------
# o config nao aceita geometria impossivel
# --------------------------------------------------------------------------


def test_mais_planos_que_regioes_e_recusado():
    with pytest.raises(ValidationError, match="regioes distintas"):
        RenderConfig(max_sub_shots=6)


def test_canvas_scale_baixo_e_recusado_com_sub_plano():
    """O quadrante sairia com metade da tela de trabalho e precisaria ser
    ampliado para preencher a saida — o oposto do que o prep existe para
    garantir. Falha no load, antes de qualquer imagem ser paga."""
    with pytest.raises(ValidationError, match="nao sustenta sub-plano"):
        RenderConfig(ken_burns={"canvas_scale": 1})


def test_canvas_scale_baixo_e_aceito_sem_sub_plano():
    cfg = RenderConfig(sub_shot_seconds=0.0, ken_burns={"canvas_scale": 1})
    assert cfg.ken_burns.canvas_scale == 1


# --------------------------------------------------------------------------
# o aviso de resolucao
# --------------------------------------------------------------------------


def test_quadrante_amplia_o_dobro_do_plano_cheio():
    """A metade dos pixels da fonte preenchendo a mesma tela. E por isso que o
    aviso mede o quadrante quando o sub-plano esta ligado: medir o plano cheio
    diria que uma imagem de 1344px esta folgada."""
    cfg = RenderConfig()
    cheio = upscale_factor(1344, "full", cfg)
    quadrante = upscale_factor(1344, "top_left", cfg)

    assert cheio == pytest.approx(1.6, abs=0.05)
    assert quadrante == pytest.approx(3.2, abs=0.05)
    assert quadrante == pytest.approx(2 * cheio)


def test_imagem_grande_nao_dispara_o_aviso():
    cfg = RenderConfig()
    assert upscale_factor(4304, "top_left", cfg) < cfg.upscale_warn_factor
