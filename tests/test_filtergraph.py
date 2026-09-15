"""Geracao do filtergraph. Nenhum ffmpeg e executado aqui — o que esta sob
teste e o texto do grafo e a aritmetica de tempo."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.filtergraph import (
    Chunk,
    Overlay,
    build_graph,
    build_inputs,
    frame_align,
    ken_burns_chain,
    plan_chunks,
    prep_size,
)


@pytest.fixture
def render(config):
    return config.render


def overlay(start, end, direction="zoom_in", image=Path("/img/a.png")):
    return Overlay(start=start, end=end, direction=direction, image_path=image)


# --------------------------------------------------------------------------
# a propriedade que protege o audio
# --------------------------------------------------------------------------


def test_nao_usa_xfade(render):
    """`xfade` encurta a saida em 400ms por transicao. Com ~20 transicoes o
    video sairia 8s mais curto que o audio."""
    chunk = Chunk(0, 0.0, 300.0, [overlay(30, 50), overlay(80, 100), overlay(140, 160)])
    graph = build_graph(chunk, render, Path("/w/s.ass"))
    assert "xfade" not in graph


def test_base_cobre_a_janela_inteira(render):
    """A camada base e o video original rodando de ponta a ponta: nada de
    trim, nada de concat de segmento de a-roll."""
    chunk = Chunk(0, 0.0, 300.0, [overlay(30, 50)])
    inputs = build_inputs(chunk, Path("/in.mp4"), render)
    assert inputs[:6] == ["-ss", "0.000", "-t", "300.000", "-i", "/in.mp4"]
    assert "trim" not in build_graph(chunk, render, None)


def test_crossfade_sai_do_fade_de_alpha_nas_duas_pontas(render):
    chunk = Chunk(0, 0.0, 120.0, [overlay(30, 50)])
    graph = build_graph(chunk, render, None)
    assert "fade=t=in:st=0:d=0.4:alpha=1" in graph      # a-roll -> b-roll
    assert "fade=t=out:st=19.600:d=0.4:alpha=1" in graph  # b-roll -> a-roll
    assert "format=yuva420p" in graph                    # alpha precisa existir
    assert "format=auto" in graph                        # overlay respeita o alpha


def test_overlay_e_gatilhado_na_janela_do_segmento(render):
    chunk = Chunk(0, 0.0, 120.0, [overlay(30.5, 50.25)])
    graph = build_graph(chunk, render, None)
    assert "enable='between(t,30.500,50.250)'" in graph
    assert "setpts=PTS+30.500/TB" in graph
    assert "eof_action=pass" in graph


# --------------------------------------------------------------------------
# Ken Burns
# --------------------------------------------------------------------------


def test_nao_usa_zoompan_no_motor_default(render):
    assert render.ken_burns.engine == "scale_crop"
    chunk = Chunk(0, 0.0, 120.0, [overlay(30, 50)])
    assert "zoompan" not in build_graph(chunk, render, None)


def test_zoom_in_sobe_o_zoom_ao_longo_do_segmento(render):
    chain = ken_burns_chain(overlay(0, 20, "zoom_in"), render)
    assert "eval=frame" in chain                    # zoom animado
    assert "1.000000+(0.120000)" in chain           # 1.00 -> 1.12
    assert "min(t/20.0000" in chain                 # rampa sobre a duracao
    assert "x='(iw-ow)*0.5'" in chain               # centrado


def test_zoom_out_desce_o_zoom(render):
    chain = ken_burns_chain(overlay(0, 20, "zoom_out"), render)
    assert "1.120000+(-0.120000)" in chain


def test_pan_move_a_janela_com_zoom_fixo(render):
    """Com zoom constante o tamanho do `scale` e estatico: o link nao
    reconfigura por frame, so o x do crop varia."""
    right = ken_burns_chain(overlay(0, 20, "pan_right"), render)
    left = ken_burns_chain(overlay(0, 20, "pan_left"), render)
    assert "eval=frame" not in right
    assert "scale=4300:2418" in right               # 3840*1.12, par
    assert "x='(iw-ow)*(0.000000+(1.000000)" in right   # esquerda -> direita
    assert "x='(iw-ow)*(1.000000+(-1.000000)" in left   # direita -> esquerda


def test_movimento_acontece_na_tela_de_2x(render):
    """O passo de 1px do crop tem que cair numa tela maior que a saida,
    senao vira degrau visivel depois."""
    chain = ken_burns_chain(overlay(0, 20, "zoom_in"), render)
    assert "crop=3840:2160" in chain                # 2x de 1920x1080
    assert chain.endswith("scale=1920:1080:flags=bicubic")  # downscale por ultimo


def test_prep_nunca_amplia_no_zoom_fechado(render):
    """A tela pre-renderizada tem que ser maior que a maior janela pedida."""
    width, height = prep_size(render)
    largest = render.width * render.ken_burns.canvas_scale * render.ken_burns.zoom_max
    assert width >= largest
    assert height >= render.height * render.ken_burns.canvas_scale * render.ken_burns.zoom_max


def test_direcao_desconhecida_levanta(render):
    with pytest.raises(ValueError, match="direcao de ken burns desconhecida"):
        ken_burns_chain(overlay(0, 20, "cambalhota"), render)


def test_motor_zoompan_disponivel_como_escape_hatch(config):
    render = config.render.model_copy(deep=True)
    render.ken_burns.engine = "zoompan"
    chain = ken_burns_chain(overlay(0, 20, "zoom_in"), render)
    assert "zoompan=" in chain
    assert "d=600" in chain          # 20s * 30fps
    assert "s=1920x1080" in chain


def test_zoompan_recebe_um_frame_so(config):
    """O `d` do zoompan conta frames de SAIDA por frame de ENTRADA, e a
    entrada e uma imagem em `-loop 1`. Sem o select, um segmento de 20s a
    30fps entrega 600 frames de entrada e o zoompan devolve 600 varreduras."""
    render = config.render.model_copy(deep=True)
    render.ken_burns.engine = "zoompan"
    chain = ken_burns_chain(overlay(0, 20, "zoom_in"), render)
    assert chain.startswith("select='eq(n\\,0)',zoompan=")


# --------------------------------------------------------------------------
# fallback de cor solida
# --------------------------------------------------------------------------


def test_segmento_sem_imagem_vira_cor_solida(render):
    chunk = Chunk(0, 0.0, 120.0, [Overlay(30, 50, "zoom_in", image_path=None)])
    graph = build_graph(chunk, render, None)
    inputs = build_inputs(chunk, Path("/in.mp4"), render)

    assert "zoompan" not in graph and "eval=frame" not in graph
    assert "lavfi" in inputs
    assert f"color=c={render.solid_fallback_color}:s=1920x1080:r=30" in inputs
    # ainda recebe os fades, para nao aparecer de estalo
    assert "fade=t=in:st=0:d=0.4:alpha=1" in graph


# --------------------------------------------------------------------------
# entradas e labels
# --------------------------------------------------------------------------


def test_ordem_das_entradas_casa_com_os_labels(render):
    overlays = [overlay(30, 50), Overlay(80, 100, "pan_right", None), overlay(140, 160)]
    chunk = Chunk(0, 0.0, 200.0, overlays)
    graph = build_graph(chunk, render, None)
    inputs = build_inputs(chunk, Path("/in.mp4"), render)

    # entrada 0 e o video base, 1..3 sao os overlays na mesma ordem
    assert inputs.count("-i") == 4
    for position in (1, 2, 3):
        assert f"[{position}:v]" in graph
    assert graph.index("[1:v]") < graph.index("[2:v]") < graph.index("[3:v]")


def test_overlays_encadeiam_em_serie(render):
    chunk = Chunk(0, 0.0, 200.0, [overlay(30, 50), overlay(80, 100)])
    graph = build_graph(chunk, render, None)
    assert "[base0][b1]overlay" in graph
    assert "[base1][b2]overlay" in graph
    assert "[base2]" in graph


def test_legenda_e_queimada_no_fim_da_cadeia(render, tmp_path):
    subs = tmp_path / "subs.ass"
    chunk = Chunk(0, 0.0, 200.0, [overlay(30, 50)])
    graph = build_graph(chunk, render, subs)
    assert graph.rstrip().endswith("[vout]")
    assert "subtitles=" in graph.splitlines()[-1]
    # a legenda entra depois dos overlays, senao o b-roll cobriria o texto
    assert graph.index("overlay") < graph.index("subtitles=")


def test_caminho_da_legenda_e_escapado(render):
    chunk = Chunk(0, 0.0, 200.0, [])
    graph = build_graph(chunk, render, Path("/w/my:video/subs.ass"))
    assert r"my\:video" in graph


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------


def test_grafo_pequeno_fica_em_um_chunk(render):
    chunks = plan_chunks([overlay(30, 50), overlay(80, 100)], 200.0, render)
    assert len(chunks) == 1
    assert chunks[0].start == 0.0 and chunks[0].end == 200.0


def test_grafo_grande_e_quebrado_em_chunks(render):
    overlays = [overlay(30 + i * 40, 50 + i * 40) for i in range(20)]
    chunks = plan_chunks(overlays, 900.0, render)

    assert len(chunks) > 1
    for chunk in chunks:
        graph = build_graph(chunk, render, Path("/w/s.ass"))
        assert len(graph) <= render.max_filtergraph_chars, f"chunk {chunk.index}: {len(graph)}"


def test_chunks_cobrem_a_timeline_sem_buraco_nem_sobreposicao(render):
    """Se as duracoes dos chunks nao somarem exatamente a duracao original,
    o audio (que e stream copy) deriva na concatenacao."""
    overlays = [overlay(30 + i * 40, 50 + i * 40) for i in range(20)]
    chunks = plan_chunks(overlays, 900.0, render)

    assert chunks[0].start == 0.0
    assert chunks[-1].end == 900.0
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.end == nxt.start
    assert sum(c.duration for c in chunks) == pytest.approx(900.0)


def test_corte_cai_em_aroll_puro(render):
    """Os dois fades vivem dentro do segmento de b-roll, entao cortar no
    inicio de um b-roll nunca parte uma transicao no meio."""
    overlays = [overlay(30 + i * 40, 50 + i * 40) for i in range(20)]
    starts = {o.start for o in overlays}
    chunks = plan_chunks(overlays, 900.0, render)
    for chunk in chunks[1:]:
        assert chunk.start in starts


def test_overlays_ficam_em_tempo_local_do_chunk(render):
    overlays = [overlay(30 + i * 40, 50 + i * 40) for i in range(20)]
    chunks = plan_chunks(overlays, 900.0, render)
    for chunk in chunks:
        for local in chunk.overlays:
            assert 0.0 <= local.start < chunk.duration
            assert local.end <= chunk.duration + 1e-6


def test_todos_os_overlays_sobrevivem_ao_chunking(render):
    overlays = [overlay(30 + i * 40, 50 + i * 40) for i in range(20)]
    chunks = plan_chunks(overlays, 900.0, render)
    assert sum(len(c.overlays) for c in chunks) == len(overlays)


def test_sem_overlay_ainda_rende_um_chunk(render):
    chunks = plan_chunks([], 900.0, render)
    assert len(chunks) == 1 and chunks[0].overlays == []


# --------------------------------------------------------------------------
# alinhamento de frame
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    (0.0, 0.0),
    (1.0, 1.0),
    (1.01, 1.0),                  # 30.3 frames -> 30
    (1.017, 31 / 30),             # 30.51 frames -> 31
    (10.49, 10.5),                # 314.7 frames -> 315
])
def test_frame_align_encaixa_no_grid(value, expected):
    assert frame_align(value, 30.0) == pytest.approx(expected)


def test_frame_align_com_fps_invalido_nao_explode():
    assert frame_align(1.234, 0.0) == 1.234
