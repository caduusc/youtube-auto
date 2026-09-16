"""Estagio 3: o storyboard aprovado vira imagens, antes de a camera ligar.

Nenhum teste aqui gasta nada: o provider e o stock sao dublês. O que esta sob
teste e a chave de identidade dos itens, o portao, e o que NAO e chamado duas
vezes.
"""

from __future__ import annotations

import pytest

from test_bank_reuse import FakeEmbedder, FakeProvider

from pipeline import approval
from pipeline.approval import NotApproved
from pipeline.bank import AssetBank
from pipeline.resolver import BudgetExceeded, from_storyboard
from pipeline.schemas import Script, ScriptBeat, Storyboard, StoryboardBeat
from pipeline.stages import images
from pipeline.stages.storyboard import cache_key as storyboard_key
from pipeline.util import write_json


@pytest.fixture
def cfg(config, tmp_path):
    return config.model_copy(update={"root": tmp_path})


def um_roteiro(n=3) -> Script:
    return Script(
        slug="v1", subject="assunto", argument="argumento",
        viewer_takeaway="o que leva", duration_target_seconds=120.0,
        beats=[ScriptBeat(id=i, title=f"t{i}", text=f"texto do beat {i} " * 10)
               for i in range(1, n + 1)],
    )


def um_board(n=3, *, sub_shots=4, input_hash=None, script=None, cfg=None) -> Storyboard:
    """O storyboard com o `input_hash` que ele teria de verdade.

    O estagio compara esse hash com a chave que o storyboard teria HOJE (e a
    trava que impede gastar em cima de um roteiro editado), entao um hash
    fabricado e rejeitado — corretamente.
    """
    if input_hash is None:
        input_hash = storyboard_key(script if script is not None else um_roteiro(n), cfg)
    return Storyboard(input_hash=input_hash, beats=[
        StoryboardBeat(
            beat_id=i, script_anchor=f"texto do beat {i}",
            concept=f"a wooden workbench with tool number {i}",
            concept_tags=[f"bench{i}", "tool"], sub_shots=sub_shots,
            rationale="porque o argumento pede imagem",
            anchor_offset=0, seconds_per_shot=2.5,
        )
        for i in range(1, n + 1)
    ])


@pytest.fixture
def cenario(cfg, monkeypatch):
    """Roteiro e storyboard em disco, com o storyboard aprovado, e um provider
    falso instalado no lugar do real."""
    script = um_roteiro()
    board = um_board(script=script, cfg=cfg)
    work = cfg.work_dir / script.slug
    work.mkdir(parents=True)
    write_json(work / "storyboard.json", board)
    approval.grant(work, "storyboard", board.input_hash)

    provider = install_fakes(monkeypatch, cfg)
    return script, board, work, provider


def install_fakes(monkeypatch, cfg, *, provider=None):
    """Banco com embedder falso, provider falso, sem stock.

    O banco real carrega o `sentence-transformers` no primeiro `embed`, e
    nenhum teste pode arrastar o torch — e o estagio consulta o banco antes de
    decidir gerar, entao nao da para so trocar o provider.

    Abre um banco NOVO a cada chamada, apontando para o mesmo arquivo: e o que
    o `open_bank` real faz, e o estagio fecha o banco no fim. Reusar a mesma
    instancia fazia a segunda execucao falhar em cima de um sqlite fechado.
    """
    provider = provider if provider is not None else FakeProvider()

    def abre(config, *_a, **_k):
        # O estilo sai do config RECEBIDO, como no `open_bank` real: capturar
        # o config da fixture faria o banco continuar com o estilo antigo
        # quando o teste troca o estilo, e o isolamento por estilo — que e o
        # que decide se uma imagem pode ser reusada — nunca seria exercitado.
        return AssetBank(
            root=config.root,
            db_path=config.root / "assets" / "bank.db",
            images_dir=config.root / "assets" / "images",
            embedder=FakeEmbedder(),
            threshold=config.bank.similarity_threshold,
            style=config.style_name,
            styles=config.known_styles,
        )

    monkeypatch.setattr("pipeline.stages.images.open_bank", abre)
    monkeypatch.setattr(
        "pipeline.stages.assets.build_provider", lambda *a, **k: provider)
    monkeypatch.setattr("pipeline.stages.assets.PexelsStock", lambda *a, **k: None)
    return provider


def unit_price(cfg) -> float:
    """O preco que a estimativa usa: o do CONFIG e nao o do provider.

    A diferenca e proposital no pipeline — o `--dry-run` tem que estimar custo
    sem chave de API nenhuma — e um teste que confunde os dois mede a coisa
    errada."""
    return cfg.image_provider.replicate.cost_usd_per_image


# --------------------------------------------------------------------------
# uma imagem por beat
# --------------------------------------------------------------------------


def test_uma_imagem_por_beat_e_nao_por_sub_plano(cenario, cfg):
    """A economia inteira do desenho mora aqui: quatro sub-planos saem da
    MESMA imagem, entao tres beats de 4 planos custam tres imagens e nao
    doze."""
    script, board, work, provider = cenario
    assets = images.run(script, board, cfg)

    assert len(assets.items) == 3
    assert len(provider.calls) == 3
    assert sum(b.sub_shots for b in board.beats) == 12


def test_os_itens_sao_indexados_por_beat(cenario, cfg):
    """Antes de gravar nao existe EDL para indexar. Ver o docstring de
    `AssetItem`."""
    script, board, work, provider = cenario
    assets = images.run(script, board, cfg)

    assert sorted(i.beat_id for i in assets.items) == [1, 2, 3]
    assert all(i.segment_index == -1 for i in assets.items)


def test_o_prompt_leva_o_estilo(cenario, cfg):
    script, board, work, provider = cenario
    images.run(script, board, cfg)

    for chamada in provider.calls:
        assert cfg.style_suffix.strip() in chamada


def test_o_pedido_sai_do_storyboard_com_conceito_e_tags(cfg):
    """`from_storyboard` e o que faz o mesmo resolver servir aos dois
    caminhos — sem ele existiriam duas resolucoes divergindo em silencio."""
    pedidos = from_storyboard(um_board(2, cfg=cfg).beats)

    assert [p.key for p in pedidos] == [1, 2]
    assert all(p.keyed_by == "beat" for p in pedidos)
    assert pedidos[0].concept_tags == ["bench1", "tool"]


# --------------------------------------------------------------------------
# o portao
# --------------------------------------------------------------------------


def test_nao_gasta_sem_o_storyboard_aprovado(cfg, monkeypatch):
    script = um_roteiro()
    board = um_board(script=script, cfg=cfg)
    work = cfg.work_dir / script.slug
    work.mkdir(parents=True)
    write_json(work / "storyboard.json", board)

    provider = install_fakes(monkeypatch, cfg)

    with pytest.raises(NotApproved, match="pipeline approve storyboard v1"):
        images.run(script, board, cfg)
    assert provider.calls == []


def test_storyboard_editado_depois_de_aprovar_revoga(cenario, cfg):
    script, board, work, provider = cenario
    outro = um_board(input_hash="board-2")

    with pytest.raises(NotApproved, match="mudou depois de aprovado"):
        images.run(script, outro, cfg)
    assert provider.calls == []


# --------------------------------------------------------------------------
# cache: nao pagar duas vezes pela mesma coisa
# --------------------------------------------------------------------------


def test_segunda_chamada_nao_gera_de_novo(cenario, cfg):
    script, board, work, provider = cenario
    images.run(script, board, cfg)
    images.run(script, board, cfg)

    assert len(provider.calls) == 3, "as imagens foram pagas duas vezes"


def test_trocar_de_estilo_nao_reusa_a_imagem_do_outro(cenario, cfg):
    """O campo mais importante do config, e duas defesas em serie.

    A chave de cache do estagio muda, entao ele roda de novo; e o banco isola
    por estilo, entao as imagens do estilo anterior nao sao candidatas a
    reuso. Sem a segunda defesa a primeira nao serviria de nada: o estagio
    rodaria e devolveria as MESMAS imagens, agora vindas do banco.

    Troca o `style` E o `style_suffix`, que e o que editar o config faz de
    verdade — o `style_name`, que e a identidade que o banco guarda, sai do
    `style`. Trocar so o sufixo faz o estagio rodar de novo e o banco reusar
    tudo (medido), e e uma inconsistencia real: ver o relato na PR.
    """
    script, board, work, provider = cenario
    images.run(script, board, cfg)

    outro = cfg.model_copy(deep=True)
    outro.style = "editorial_warm"
    outro.style_suffix = cfg.styles["editorial_warm"]
    # o estilo entra na chave do storyboard tambem, entao o estagio 2 o
    # refaria: aqui o board e recalculado para o config novo, com os mesmos
    # conceitos, que e o caso interessante — o banco e que decide reusar ou nao
    board_novo = um_board(script=script, cfg=outro)
    write_json(work / "storyboard.json", board_novo)
    approval.grant(work, "storyboard", board_novo.input_hash)
    assets = images.run(script, board_novo, outro)

    assert len(provider.calls) == 6
    assert assets.by_origin == {"generated": 3}


def test_a_chave_inclui_o_storyboard(cenario, cfg):
    script, board, work, provider = cenario
    assert images.cache_key(board, cfg) != images.cache_key(
        um_board(input_hash="outro"), cfg)


# --------------------------------------------------------------------------
# dry-run e orcamento
# --------------------------------------------------------------------------


def test_dry_run_nao_gasta_e_escreve_em_outro_arquivo(cenario, cfg):
    script, board, work, provider = cenario
    assets = images.run(script, board, cfg, dry_run=True)

    assert provider.calls == []
    assert assets.total_cost_usd == 0.0
    assert (work / "assets.dryrun.json").exists()
    assert not (work / "assets.json").exists()


def test_dry_run_estima_o_que_seria_gasto(cenario, cfg):
    script, board, work, provider = cenario
    assets = images.run(script, board, cfg, dry_run=True)

    assert assets.estimate.n_broll == 3
    assert assets.estimate.n_to_generate == 3
    assert assets.estimate.worst_case_usd == pytest.approx(3 * unit_price(cfg))


def test_para_antes_de_gastar_se_o_pior_caso_estoura(cenario, cfg):
    script, board, work, provider = cenario
    apertado = cfg.model_copy(deep=True)
    # abaixo do pior caso de tres imagens, que e o que o teto compara
    apertado.budget.max_usd_per_video = 2 * unit_price(cfg)

    with pytest.raises(BudgetExceeded):
        images.run(script, board, apertado)
    assert provider.calls == [], "gastou antes de conferir o teto"


def test_o_acumulador_pega_provider_que_cobra_mais_que_o_config(cenario, cfg):
    """O pior caso passa (pelo preco do config) e o provider cobra mais caro.

    E o unico caminho em que o gasto real pode estourar um teto que a
    estimativa aprovou — provider que reajustou, cobranca por passo, retry
    cobrado duas vezes. O que sobra vira cor solida em vez de divida.
    """
    script, board, work, provider = cenario
    caro = cfg.model_copy(deep=True)
    # O config anuncia barato e o provider cobra 0.025: e o unico arranjo em
    # que o pior caso passa e o gasto real estoura. Explicito aqui para o
    # teste nao depender de qual modelo o config de exemplo usa.
    caro.image_provider.replicate.cost_usd_per_image = 0.003
    caro.budget.max_usd_per_video = 0.05          # cabem 2 a 0.025 reais

    assets = images.run(script, board, caro)

    assert assets.by_origin == {"generated": 2, "solid": 1}
    assert assets.total_cost_usd <= caro.budget.max_usd_per_video


def test_o_relatorio_mostra_o_conceito_ao_lado_do_arquivo(cenario, cfg):
    """A pergunta da revisao e se a imagem que saiu e a que foi pedida, e o
    caminho do arquivo sozinho nao responde isso."""
    script, board, work, provider = cenario
    assets = images.run(script, board, cfg)
    texto = images.render_text(assets, board, cfg)

    assert "a wooden workbench with tool number 1" in texto
    assert "beat 1" in texto
    assert "4 planos" in texto
