"""Geracao do filtergraph. Nenhum ffmpeg e executado aqui — o que esta sob
teste e o texto do grafo e a aritmetica de tempo."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.config import RenderConfig
from pipeline.filtergraph import (
    Chunk,
    Overlay,
    build_graph,
    build_inputs,
    canvas_for,
    frame_align,
    ken_burns_chain,
    plan_chunks,
    prep_size,
    region_chain,
)


@pytest.fixture
def overlay_curto_e_longo():
    return (
        Overlay(start=2.0, end=3.0, direction="zoom_in", image_path=Path("curto.png")),
        Overlay(start=5.0, end=13.0, direction="pan_right", image_path=Path("longo.png")),
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


@pytest.fixture
def chunked(render):
    """`render` com o limite baixo o bastante para os 20 overlays destes
    testes precisarem de mais de um chunk.

    O limite vive aqui, e nao herdado do config de exemplo, de proposito: o
    que estes testes cobrem e o comportamento do chunking, nao a calibracao.
    Acoplados, subir `max_filtergraph_chars` porque o sub-plano multiplicou o
    grafo por 4 fazia estes testes pararem de exercitar o caminho que eles
    existem para cobrir — e passar em silencio.
    """
    cfg = render.model_copy(deep=True)
    cfg.max_filtergraph_chars = 3000
    return cfg


def test_grafo_grande_e_quebrado_em_chunks(chunked):
    overlays = [overlay(30 + i * 40, 50 + i * 40) for i in range(20)]
    chunks = plan_chunks(overlays, 900.0, chunked)

    assert len(chunks) > 1
    for chunk in chunks:
        graph = build_graph(chunk, chunked, Path("/w/s.ass"))
        assert len(graph) <= chunked.max_filtergraph_chars, f"chunk {chunk.index}: {len(graph)}"


def test_chunks_cobrem_a_timeline_sem_buraco_nem_sobreposicao(chunked):
    """Se as duracoes dos chunks nao somarem exatamente a duracao original,
    o audio (que e stream copy) deriva na concatenacao."""
    overlays = [overlay(30 + i * 40, 50 + i * 40) for i in range(20)]
    chunks = plan_chunks(overlays, 900.0, chunked)

    assert chunks[0].start == 0.0
    assert chunks[-1].end == 900.0
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.end == nxt.start
    assert sum(c.duration for c in chunks) == pytest.approx(900.0)


def test_corte_cai_em_aroll_puro(chunked):
    """Os dois fades vivem dentro do proprio overlay, entao cortar no inicio
    de um overlay nunca parte uma transicao no meio. Com sub-planos a
    fronteira pode cair numa emenda da mesma imagem, onde o corte ja e seco —
    e por isso continua valendo."""
    overlays = [overlay(30 + i * 40, 50 + i * 40) for i in range(20)]
    starts = {o.start for o in overlays}
    chunks = plan_chunks(overlays, 900.0, chunked)
    for chunk in chunks[1:]:
        assert chunk.start in starts


def test_overlays_ficam_em_tempo_local_do_chunk(chunked):
    overlays = [overlay(30 + i * 40, 50 + i * 40) for i in range(20)]
    chunks = plan_chunks(overlays, 900.0, chunked)
    for chunk in chunks:
        for local in chunk.overlays:
            assert 0.0 <= local.start < chunk.duration
            assert local.end <= chunk.duration + 1e-6


def test_todos_os_overlays_sobrevivem_ao_chunking(chunked):
    overlays = [overlay(30 + i * 40, 50 + i * 40) for i in range(20)]
    chunks = plan_chunks(overlays, 900.0, chunked)
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


def test_chunk_que_nao_cabe_avisa(capsys):
    """Com aperto de pausa, o grafo cresce com a duracao e nao com os b-rolls.

    As fronteiras de chunk sao as de b-roll, entao a-roll corrido com muitas
    emendas nao tem onde ser partido. O aviso e a unica saida honesta: nao
    existe fallback bom — partir no meio de uma emenda mudaria o corte.
    """
    cfg = RenderConfig(max_filtergraph_chars=400)
    keep = [(i * 0.6, i * 0.6 + 0.4) for i in range(40)]
    chunks = plan_chunks([], 24.0, cfg, window=lambda a, b: (a, b - a, keep))

    assert len(chunks) == 1                      # sem b-roll, nao ha onde partir
    saida = capsys.readouterr().out
    assert "render.warn" in saida
    assert "emendas=40" in saida
    assert "pause_max_seconds" in saida


def test_chunk_que_cabe_nao_avisa(capsys):
    chunks = plan_chunks([], 24.0, RenderConfig(), window=lambda a, b: (a, b - a, [(0.0, 24.0)]))
    assert len(chunks) == 1
    assert "render.warn" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# fade proporcional: o que torna b-roll curto representavel
# --------------------------------------------------------------------------


def test_fade_fixo_em_segmento_longo():
    """Acima de 2x o crossfade nada muda, entao config antigo nao regride."""
    from pipeline.filtergraph import fade_for

    cfg = RenderConfig(crossfade_seconds=0.4)
    for duracao in (2.0, 6.0, 25.0):
        assert fade_for(duracao, cfg) == pytest.approx(0.4)


def test_fade_encolhe_com_segmento_curto():
    """Com fade fixo, 0.8s gastaria 0.4s entrando e 0.4s saindo: a imagem
    apareceria como um triangulo e nunca chegaria a opacidade cheia."""
    from pipeline.filtergraph import fade_for

    cfg = RenderConfig(crossfade_seconds=0.4, max_fade_ratio=0.25)
    for duracao in (0.8, 1.0, 1.5):
        fade = fade_for(duracao, cfg)
        cheio = duracao - 2 * fade
        assert cheio == pytest.approx(duracao * 0.5), duracao
        assert cheio > 0


def test_fade_de_duracao_zero_nao_e_negativo():
    from pipeline.filtergraph import fade_for

    assert fade_for(0.0, RenderConfig()) == 0.0
    assert fade_for(-1.0, RenderConfig()) == 0.0


def test_grafo_usa_o_fade_do_segmento(render, overlay_curto_e_longo):
    """Dois b-rolls de duracoes diferentes no MESMO grafo usam fades
    diferentes — o valor nao pode ser calculado uma vez fora do laco."""
    curto, longo = overlay_curto_e_longo
    graph = build_graph(
        Chunk(0, 0.0, 30.0, [curto, longo], keep=[(0.0, 30.0)],
              input_start=0.0, input_duration=30.0),
        render, None,
    )
    assert "d=0.25:alpha=1" in graph      # o de 1.0s
    assert "d=0.4:alpha=1" in graph       # o de 8.0s


# --------------------------------------------------------------------------
# sub-planos: a geometria
# --------------------------------------------------------------------------


def test_prep_e_multiplo_de_quatro(render):
    """Para que o quadrante — metade do prep — caia em pixel par sem o crop
    precisar arredondar. Sem isso, 1920x1080 daria prep 4302 e quadrante
    2151, impar, e arredondar para baixo comeria a margem de 0.6px que
    existe contra ampliar."""
    width, height = prep_size(render)
    assert width % 4 == 0 and height % 4 == 0


@pytest.mark.parametrize("region", ["full", "top_left", "top_right",
                                    "bottom_left", "bottom_right"])
def test_nenhuma_regiao_amplia(render, region):
    """A propriedade que torna o sub-plano possivel: o `scale` animado de
    QUALQUER regiao ainda reduz. E o que o quadrante gasta inteiro — ele bate
    no limite exatamente, por construcao."""
    prep_w, prep_h = prep_size(render)
    fonte_w = prep_w if region == "full" else prep_w // 2
    fonte_h = prep_h if region == "full" else prep_h // 2

    canvas_w, canvas_h = canvas_for(region, render)
    pedido_w = canvas_w * render.ken_burns.zoom_max
    pedido_h = canvas_h * render.ken_burns.zoom_max

    assert fonte_w >= pedido_w, f"{region}: {fonte_w}px para {pedido_w:.1f}px pedidos"
    assert fonte_h >= pedido_h, f"{region}: {fonte_h}px para {pedido_h:.1f}px pedidos"


def test_um_terco_ampliaria(render):
    """Por que o teto e o quadrante e nao uma grade mais fina. Um terco da
    tela preparada nao alcanca o que o Ken Burns pede — e por isso que
    `SUB_SHOT_ORDER` tem cinco regioes e nao dez."""
    prep_w, _ = prep_size(render)
    pedido = render.width * render.ken_burns.zoom_max
    assert prep_w // 2 >= pedido
    assert prep_w // 3 < pedido


def test_quadrante_trabalha_na_resolucao_de_saida(render):
    """A consequencia honesta de `canvas_scale: 2`: no plano cheio o passo de
    1px do crop cai numa tela 2x e vira meio pixel, no quadrante a tela de
    trabalho JA e a saida."""
    assert canvas_for("full", render) == (3840, 2160)
    assert canvas_for("top_left", render) == (1920, 1080)


@pytest.mark.parametrize("region,esperado", [
    ("top_left", "crop=iw/2:ih/2:0:0,"),
    ("top_right", "crop=iw/2:ih/2:(iw-ow):0,"),
    ("bottom_left", "crop=iw/2:ih/2:0:(ih-oh),"),
    ("bottom_right", "crop=iw/2:ih/2:(iw-ow):(ih-oh),"),
])
def test_cada_quadrante_recorta_o_seu_canto(region, esperado):
    assert region_chain(region) == esperado


def test_plano_cheio_nao_recorta_nada():
    assert region_chain("full") == ""


def test_regiao_desconhecida_levanta():
    with pytest.raises(ValueError, match="regiao de sub-plano desconhecida"):
        region_chain("meio")


def test_recorte_da_regiao_vem_antes_do_ken_burns(render):
    """A ordem nao e cosmetica: recortar depois do `scale` animado
    recortaria de uma tela que muda de tamanho a cada frame, e a regiao
    andaria junto com o zoom."""
    chain = ken_burns_chain(overlay(0, 2.5, "zoom_in"), render)
    cheio = ken_burns_chain(Overlay(0, 2.5, "zoom_in", Path("/img/a.png"), "full"), render)
    quadrante = ken_burns_chain(
        Overlay(0, 2.5, "zoom_in", Path("/img/a.png"), "bottom_left"), render)

    assert chain == cheio                      # `full` e o default
    assert quadrante.startswith("crop=iw/2:ih/2:0:(ih-oh),scale=")


def test_zoompan_recorta_depois_do_select(config):
    """Antes do select, o crop rodaria em cada frame do `-loop 1` para ser
    descartado em seguida."""
    render = config.render.model_copy(deep=True)
    render.ken_burns.engine = "zoompan"
    chain = ken_burns_chain(
        Overlay(0, 2.5, "zoom_in", Path("/img/a.png"), "top_right"), render)
    assert chain.startswith("select='eq(n\\,0)',crop=iw/2:ih/2:(iw-ow):0,zoompan=")


# --------------------------------------------------------------------------
# sub-planos: os fades da emenda
# --------------------------------------------------------------------------


def _fades(graph: str, label: str) -> list[str]:
    """Os filtros de fade da cadeia daquele label."""
    trecho = next(p for p in graph.split(";\n") if p.endswith(f"[{label}]"))
    return [f for f in trecho.split(",") if f.startswith("fade=")]


def test_emenda_entre_sub_planos_nao_tem_fade(render):
    """A propriedade que o sub-plano existe para nao quebrar: um fade no meio
    da faixa faria a imagem DO USUARIO reaparecer por 400ms entre dois planos
    da mesma foto. So a primeira ponta entra e so a ultima sai."""
    planos = [
        Overlay(0.0, 2.5, "zoom_in", Path("/i.png"), "full", fade_in=0.4, fade_out=0.0),
        Overlay(2.5, 5.0, "pan_right", Path("/i.png"), "top_left", fade_in=0.0, fade_out=0.0),
        Overlay(5.0, 7.5, "zoom_out", Path("/i.png"), "bottom_right", fade_in=0.0, fade_out=0.4),
    ]
    graph = build_graph(Chunk(0, 0.0, 30.0, planos), render, None)

    assert _fades(graph, "b1") == ["fade=t=in:st=0:d=0.4:alpha=1"]
    assert _fades(graph, "b2") == []
    assert _fades(graph, "b3") == ["fade=t=out:st=2.100:d=0.4:alpha=1"]


def test_fade_none_continua_derivando_da_duracao(render):
    """O caminho de sempre, de um plano so por faixa, nao muda."""
    graph = build_graph(Chunk(0, 0.0, 30.0, [overlay(2.0, 3.0)]), render, None)
    assert _fades(graph, "b1") == [
        "fade=t=in:st=0:d=0.25:alpha=1", "fade=t=out:st=0.750:d=0.25:alpha=1",
    ]


def test_regiao_e_fade_sobrevivem_a_divisao_em_chunks(render):
    """`_localize` reescreve os tempos; se ele perder a regiao, todo plano de
    um video longo volta a ser o plano cheio — e nada falharia."""
    render = render.model_copy(deep=True)
    render.max_filtergraph_chars = 600
    planos = [
        Overlay(10.0, 12.5, "zoom_in", Path("/i.png"), "full", fade_in=0.4, fade_out=0.0),
        Overlay(12.5, 15.0, "pan_right", Path("/i.png"), "bottom_right", 0.0, 0.0),
        Overlay(60.0, 62.5, "zoom_out", Path("/i.png"), "top_right", 0.0, 0.4),
    ]
    chunks = plan_chunks(planos, 120.0, render)
    assert len(chunks) > 1

    vistos = [(o.region, o.fade_in, o.fade_out) for c in chunks for o in c.overlays]
    assert vistos == [("full", 0.4, 0.0), ("bottom_right", 0.0, 0.0),
                      ("top_right", 0.0, 0.4)]
