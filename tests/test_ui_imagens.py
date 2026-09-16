"""A tela de revisao das imagens: contact sheet, regerar, aprovar.

O que esta sob teste e o que custa dinheiro e o que cobre o que. Nenhum teste
aqui chama provider de verdade, e um deles existe so para provar que ABRIR a
pagina nao chama.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient
from test_images_stage import install_fakes, um_board, um_roteiro

from pipeline import approval, progress
from pipeline.schemas import Assets, Storyboard
from pipeline.script import render as render_script
from pipeline.stages import images
from pipeline.ui import create_app
from pipeline.util import read_json, write_json


@pytest.fixture
def cfg(config, tmp_path):
    return config.model_copy(update={"root": tmp_path})


@pytest.fixture
def projeto(cfg, monkeypatch):
    """Roteiro, storyboard aprovado e as tres imagens ja resolvidas."""
    script = um_roteiro()
    board = um_board(script=script, cfg=cfg)
    work = cfg.work_dir / script.slug
    work.mkdir(parents=True)
    (work / "script.md").write_text(render_script(script), encoding="utf-8")
    write_json(work / "storyboard.json", board)
    approval.grant(work, "storyboard", board.input_hash)

    provider = install_fakes(monkeypatch, cfg)
    assets = images.run(script, board, cfg)
    return script, board, work, provider, assets


@pytest.fixture
def client(cfg):
    return TestClient(create_app(cfg))


# --------------------------------------------------------------------------
# a folha de contato
# --------------------------------------------------------------------------


def test_sem_imagens_diz_qual_comando_rodar(client, cfg):
    (cfg.work_dir / "vazio").mkdir(parents=True)
    corpo = client.get("/vazio/assets").text

    assert "ainda nao tem imagens" in corpo
    assert "pipeline images vazio" in corpo


def test_abrir_a_pagina_nao_gera_imagem(client, projeto):
    """Navegador recarrega por conta propria e faz prefetch de link. Uma
    pagina que gera imagem ao abrir e uma armadilha caríssima."""
    script, board, work, provider, assets = projeto
    antes = len(provider.calls)

    client.get("/v1/assets")
    client.get("/v1/assets")

    assert len(provider.calls) == antes


def test_a_folha_mostra_conceito_custo_e_origem(client, projeto):
    script, board, work, provider, assets = projeto
    corpo = client.get("/v1/assets").text

    assert "beat 1" in corpo and "beat 3" in corpo
    assert "a wooden workbench with tool number 1" in corpo
    assert "generated" in corpo
    assert "4 planos de 2.5s" in corpo


def test_a_miniatura_aponta_para_a_rota_da_imagem(client, projeto):
    script, board, work, provider, assets = projeto
    corpo = client.get("/v1/assets").text
    assert '<img src="/v1/assets/1/image"' in corpo


def test_a_rota_serve_o_arquivo(client, projeto, cfg):
    script, board, work, provider, assets = projeto
    item = next(i for i in assets.items if i.beat_id == 1)

    resposta = client.get("/v1/assets/1/image")

    assert resposta.status_code == 200
    assert resposta.content == cfg.path(item.path).read_bytes()


def test_a_rota_recusa_caminho_fora_do_diretorio_de_imagens(client, projeto, cfg):
    """`assets.json` e escrito pelo pipeline e nao por voce, mas servir um
    caminho de dentro de um arquivo sem conferir envelhece mal."""
    script, board, work, provider, assets = projeto
    fora = cfg.root / "segredo.txt"
    fora.write_text("nao e imagem", encoding="utf-8")

    adulterado = assets.model_copy(update={
        "items": [i.model_copy(update={"path": "segredo.txt"}) if i.beat_id == 1 else i
                  for i in assets.items]})
    write_json(work / "assets.json", adulterado)

    assert client.get("/v1/assets/1/image").status_code == 404


def test_beat_sem_imagem_mostra_cor_solida(client, projeto):
    script, board, work, provider, assets = projeto
    solido = assets.model_copy(update={
        "items": [i.model_copy(update={"path": None, "origin": "solid"})
                  if i.beat_id == 2 else i for i in assets.items]})
    write_json(work / "assets.json", solido)

    corpo = client.get("/v1/assets").text
    assert "cor solida" in corpo


def test_o_aviso_de_resolucao_aparece_por_imagem(client, projeto):
    """As imagens do dublê tem 8 bytes e nao sao imagem: e o caso do arquivo
    ilegivel, que a tela tem que nomear em vez de mostrar quadro vazio."""
    script, board, work, provider, assets = projeto
    corpo = client.get("/v1/assets").text
    assert "ilegivel" in corpo


# --------------------------------------------------------------------------
# aprovar
# --------------------------------------------------------------------------


def test_aprovar_grava_o_digest_do_conteudo(client, projeto):
    script, board, work, provider, assets = projeto
    client.post("/v1/assets/approve")

    guardado = approval.read(work, "images")
    assert guardado is not None and guardado.matches(assets.digest())
    assert progress.images_gate(work).state == progress.APPROVED


def test_o_botao_de_aprovar_sai_quando_nao_ha_o_que_aprovar(client, projeto):
    script, board, work, provider, assets = projeto
    assert "Aprovar" in client.get("/v1/assets").text

    client.post("/v1/assets/approve")
    corpo = client.get("/v1/assets").text
    assert "Aprovar" not in corpo


# --------------------------------------------------------------------------
# regerar
# --------------------------------------------------------------------------


def test_regerar_gasta_e_troca_so_aquela_imagem(client, projeto):
    script, board, work, provider, assets = projeto
    antes = {i.beat_id: i.path for i in assets.items}

    client.post("/v1/assets/2/regenerate")

    depois = read_json(work / "assets.json", Assets)
    novo = {i.beat_id: i.path for i in depois.items}
    assert novo[2] != antes[2], "a imagem do beat 2 nao mudou"
    assert novo[1] == antes[1] and novo[3] == antes[3], "mexeu nas outras"
    assert len(provider.calls) == 4


def test_regerar_revoga_a_aprovacao(client, projeto):
    """Voce aprovou um conjunto de imagens; este e outro. Sem isso, a
    aprovacao guardada em cima do `input_hash` cobriria em silencio uma imagem
    que voce nunca viu — o `input_hash` nao muda, porque o storyboard nao
    mudou."""
    script, board, work, provider, assets = projeto
    client.post("/v1/assets/approve")
    assert progress.images_gate(work).state == progress.APPROVED

    client.post("/v1/assets/2/regenerate")

    assert progress.images_gate(work).state == progress.EDITED
    assert "Aprovar de novo" in client.get("/v1/assets").text


def test_regerar_nao_consulta_o_banco(client, projeto):
    """A imagem recusada acabou de entrar no banco com exatamente este
    conceito: uma resolucao normal a devolveria de volta com similaridade
    1.0, e o botao nao faria nada visivel."""
    script, board, work, provider, assets = projeto
    antigo = next(i for i in assets.items if i.beat_id == 2)

    client.post("/v1/assets/2/regenerate")
    depois = read_json(work / "assets.json", Assets)
    novo = next(i for i in depois.items if i.beat_id == 2)

    assert novo.origin == "generated"
    assert novo.path != antigo.path
    assert novo.cost_usd > 0


def test_regerar_apaga_a_imagem_recusada_do_banco(client, projeto, cfg):
    """Deixar a linha la faz o proximo video com conceito parecido reusar
    exatamente a imagem que voce rejeitou, e em silencio — reuso do banco nao
    passa por aprovacao nenhuma."""
    script, board, work, provider, assets = projeto
    antigo = next(i for i in assets.items if i.beat_id == 2)
    arquivo = cfg.path(antigo.path)
    assert arquivo.exists()

    client.post("/v1/assets/2/regenerate")

    assert not arquivo.exists()
    from pipeline.bank import AssetBank
    from test_bank_reuse import FakeEmbedder
    bank = AssetBank(root=cfg.root, db_path=cfg.root / "assets" / "bank.db",
                     images_dir=cfg.root / "assets" / "images",
                     embedder=FakeEmbedder(),
                     threshold=cfg.bank.similarity_threshold,
                     style=cfg.style_name, styles=cfg.known_styles)
    try:
        ids = {row["id"] for row in bank.conn.execute("SELECT id FROM assets")}
    finally:
        bank.close()
    assert antigo.asset_id not in ids


def test_regerar_respeita_o_teto_do_orcamento(client, projeto, cfg):
    """O teto e por video, e regerar soma em cima do que ja foi gasto."""
    script, board, work, provider, assets = projeto
    apertado = cfg.model_copy(deep=True)
    apertado.budget.max_usd_per_video = assets.total_cost_usd

    apertado_client = TestClient(create_app(apertado))
    antes = len(provider.calls)
    resposta = apertado_client.post("/v1/assets/2/regenerate")

    assert len(provider.calls) == antes, "gastou acima do teto"
    assert "teto" in resposta.text or "acima" in resposta.text


def test_regerar_beat_que_nao_existe_nao_derruba_a_pagina(client, projeto):
    resposta = client.post("/v1/assets/99/regenerate")

    assert resposta.status_code == 200
    assert "beat 99" in resposta.text


def test_o_storyboard_linka_para_as_imagens(client, projeto):
    script, board, work, provider, assets = projeto
    assert 'href="/v1/assets"' in client.get("/v1/storyboard").text
