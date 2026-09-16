"""A UI de revisao e aprovacao.

Nenhum teste aqui sobe servidor de verdade nem toca em rede: o `TestClient` do
starlette chama a aplicacao em processo. O que esta sob teste e o que a UI faz
com o DISCO — o que ela grava, o que ela se recusa a gravar, e o que ela NAO
chama.
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from pipeline import approval, progress, stages
from pipeline.schemas import Storyboard, StoryboardBeat
from pipeline.script import read as read_script
from pipeline.ui import NotFound, anchor_markup, create_app, work_for
from pipeline.util import write_json

ROTEIRO = """\
---
slug: v1
subject: como montar uma esteira de video
argument: a automacao ja e barata o bastante
viewer_takeaway: da para montar o estudio em software
duration_target_seconds: 60.0
---

## beat 1 — abertura
Todo mundo acha que precisa de uma equipe para isso.

## beat 2 — o problema
O custo nao esta na camera, esta no tempo de edicao.
"""


@pytest.fixture
def cfg(config, tmp_path):
    return config.model_copy(update={"root": tmp_path})


@pytest.fixture
def work(cfg):
    caminho = cfg.work_dir / "v1"
    caminho.mkdir(parents=True)
    (caminho / "script.md").write_text(ROTEIRO, encoding="utf-8")
    return caminho


@pytest.fixture
def client(cfg):
    return TestClient(create_app(cfg))


def um_board(work, *, anchor="O custo nao esta na camera", input_hash="h1") -> Storyboard:
    board = Storyboard(input_hash=input_hash, beats=[
        StoryboardBeat(
            beat_id=2, script_anchor=anchor,
            concept="a workshop bench with a single lamp",
            concept_tags=["bench", "lamp"], sub_shots=3,
            rationale="o argumento precisa de imagem concreta",
            anchor_offset=0, seconds_per_shot=2.5,
        ),
    ])
    write_json(work / "storyboard.json", board)
    return board


# --------------------------------------------------------------------------
# indice
# --------------------------------------------------------------------------


def test_indice_vazio_diz_por_onde_comecar(client):
    resposta = client.get("/")
    assert resposta.status_code == 200
    assert "Nenhum video" in resposta.text
    assert "pipeline script" in resposta.text


def test_indice_mostra_o_estado_de_cada_portao(client, work):
    corpo = client.get("/").text
    assert "v1" in corpo
    assert progress.WRITTEN in corpo          # roteiro escrito, nao aprovado
    assert "pronto para gravar" not in corpo


def test_indice_marca_editado_depois_de_aprovar_e_mexer(client, work):
    approval.grant(work, "script", read_script(work / "script.md").digest())
    assert progress.APPROVED in client.get("/").text

    (work / "script.md").write_text(
        ROTEIRO.replace("no tempo de edicao", "no tempo que a edicao come"),
        encoding="utf-8")
    assert progress.EDITED in client.get("/").text


def test_roteiro_ilegivel_aparece_no_indice_com_o_erro(client, cfg):
    quebrado = cfg.work_dir / "v2"
    quebrado.mkdir(parents=True)
    (quebrado / "script.md").write_text("## beat 1\nsem frontmatter\n", encoding="utf-8")

    corpo = client.get("/").text
    assert "v2" in corpo
    assert "frontmatter" in corpo


# --------------------------------------------------------------------------
# slug
# --------------------------------------------------------------------------


def test_slug_que_nao_existe_da_404_com_a_lista(client, work):
    resposta = client.get("/v9/script")
    assert resposta.status_code == 404
    assert "v1" in resposta.text


@pytest.mark.parametrize("slug", [
    "../fora", "..", "../../etc", "v1/../../fora", "..%2Ffora", "/etc",
])
def test_slug_nao_escapa_do_work_dir(cfg, work, tmp_path, slug):
    """A UI ESCREVE arquivo (o roteiro), entao a diferenca importa mesmo
    servindo em localhost.

    Testa `work_for` direto e nao por HTTP de proposito: o cliente normaliza
    `../` antes de mandar, entao um `POST /../fora/script` nunca chega nesta
    funcao — a primeira versao deste teste passava sem exercitar a validacao,
    e continuava passando com a validacao trocada por `work_dir / slug`.
    """
    (tmp_path / "fora").mkdir(exist_ok=True)      # existe, para nao passar por acaso

    with pytest.raises(NotFound):
        work_for(cfg, slug)


def test_slug_que_existe_resolve_dentro_do_work_dir(cfg, work):
    assert work_for(cfg, "v1") == work
    assert work_for(cfg, "v1").parent == cfg.work_dir


# --------------------------------------------------------------------------
# roteiro
# --------------------------------------------------------------------------


def test_pagina_do_roteiro_mostra_o_cabecalho_e_os_beats(client, work):
    corpo = client.get("/v1/script").text
    assert "como montar uma esteira de video" in corpo
    assert "Todo mundo acha que precisa de uma equipe" in corpo
    assert "2 beats" in corpo
    assert "<textarea" in corpo


def test_video_sem_roteiro_diz_qual_comando_rodar(client, cfg):
    (cfg.work_dir / "v3").mkdir(parents=True)
    corpo = client.get("/v3/script").text
    assert "ainda nao tem roteiro" in corpo
    assert "pipeline script" in corpo


def test_salvar_grava_verbatim(client, work):
    """Nao o `render()` do que foi parseado.

    O render normaliza: ele descartaria a nota antes do primeiro beat, a ordem
    do frontmatter e o espacamento. Parsear garante que o proximo estagio le o
    arquivo; nao autoriza reescrever o que a pessoa digitou.
    """
    texto = ROTEIRO.replace(
        "## beat 1", "<!-- lembrar de regravar a abertura -->\n\n## beat 1")
    resposta = client.post("/v1/script", data={"texto": texto})

    assert resposta.status_code == 200          # seguiu o redirect 303
    assert (work / "script.md").read_text(encoding="utf-8") == texto


def test_salvar_texto_quebrado_nao_toca_no_arquivo(client, work):
    antes = (work / "script.md").read_text(encoding="utf-8")
    resposta = client.post("/v1/script", data={"texto": "## beat 1\nsem frontmatter\n"})

    assert (work / "script.md").read_text(encoding="utf-8") == antes
    assert "frontmatter" in resposta.text
    # e o texto da pessoa volta no campo, para ela consertar em vez de redigitar
    assert "sem frontmatter" in resposta.text


def test_salvar_normaliza_crlf(client, work):
    """O textarea devolve CRLF. Sem normalizar, o `\\r` sobrevive DENTRO do
    texto do beat — o parser junta linhas com "\\n" e so tira espaco das
    pontas — e entra no prompt e no digest."""
    client.post("/v1/script", data={"texto": ROTEIRO.replace("\n", "\r\n")})

    gravado = (work / "script.md").read_text(encoding="utf-8")
    assert "\r" not in gravado
    assert all("\r" not in b.text for b in read_script(work / "script.md").beats)


def test_texto_do_roteiro_sai_escapado(client, work):
    (work / "script.md").write_text(
        ROTEIRO.replace("uma equipe", "uma <script>alert(1)</script> equipe"),
        encoding="utf-8")
    corpo = client.get("/v1/script").text

    assert "<script>alert(1)</script>" not in corpo
    assert "&lt;script&gt;" in corpo


# --------------------------------------------------------------------------
# aprovacao
# --------------------------------------------------------------------------


def test_aprovar_roteiro_grava_o_digest_do_que_esta_na_tela(client, work):
    client.post("/v1/script/approve")

    guardado = approval.read(work, "script")
    assert guardado is not None
    assert guardado.matches(read_script(work / "script.md").digest())


def test_aprovar_e_depois_editar_revoga(client, work):
    client.post("/v1/script/approve")
    client.post("/v1/script", data={"texto": ROTEIRO.replace("camera", "lente")})

    assert progress.script_gate(work).state == progress.EDITED
    assert "Aprovar de novo" in client.get("/v1/script").text


def test_botao_de_aprovar_nao_aparece_quando_nao_ha_o_que_aprovar(client, work):
    client.post("/v1/script/approve")
    assert "Aprovar" not in client.get("/v1/script").text


# --------------------------------------------------------------------------
# storyboard
# --------------------------------------------------------------------------


def test_storyboard_ausente_nao_chama_o_agente(client, work, monkeypatch):
    """Um GET que gasta uma chamada de Opus seria armadilha: navegador
    recarrega por conta propria, e prefetch de link tambem."""
    def explode(*a, **k):
        raise AssertionError("a UI chamou o estagio do storyboard num GET")

    monkeypatch.setattr(stages.storyboard, "run", explode)

    corpo = client.get("/v1/storyboard").text
    assert "pipeline storyboard v1" in corpo
    assert "nao roda o agente" in corpo


def test_storyboard_mostra_um_cartao_por_beat(client, work):
    um_board(work)
    corpo = client.get("/v1/storyboard").text

    assert "beat 2" in corpo
    assert "a workshop bench with a single lamp" in corpo
    assert "o argumento precisa de imagem concreta" in corpo
    assert "3 planos de 2.5s = 7.5s na tela" in corpo


def test_storyboard_destaca_o_ancora_dentro_da_fala(client, work):
    um_board(work)
    corpo = client.get("/v1/storyboard").text
    assert "<mark>O custo nao esta na camera</mark>" in corpo


def test_aprovar_storyboard_usa_o_hash_do_disco_sem_chamar_o_agente(
        client, work, monkeypatch):
    board = um_board(work)

    def explode(*a, **k):
        raise AssertionError("aprovar chamou o estagio e podia gastar")

    monkeypatch.setattr(stages.storyboard, "run", explode)
    client.post("/v1/storyboard/approve")

    guardado = approval.read(work, "storyboard")
    assert guardado is not None and guardado.matches(board.input_hash)


def test_storyboard_de_outra_versao_do_config_aparece_como_editado(client, work):
    """O `input_hash` do storyboard inclui as regras editoriais, o
    `sub_shot_seconds` e o estilo. Mexer no config invalida o artefato, e sem
    comparar o digest a tela diria "aprovado" para algo que o proximo estagio
    vai recalcular."""
    um_board(work, input_hash="h1")
    client.post("/v1/storyboard/approve")
    assert progress.storyboard_gate(work).state == progress.APPROVED

    um_board(work, input_hash="h2")
    assert progress.storyboard_gate(work).state == progress.EDITED
    assert progress.EDITED in client.get("/v1/storyboard").text


def test_storyboard_ilegivel_nao_derruba_a_pagina(client, work):
    (work / "storyboard.json").write_text("{isto nao e json", encoding="utf-8")
    resposta = client.get("/v1/storyboard")

    assert resposta.status_code == 200
    assert "nao pode ser lido" in resposta.text


def test_beat_que_saiu_do_roteiro_e_apontado(client, work):
    """Apagar o beat 2 do roteiro deixa a imagem dele sem lugar. A tela tem
    que dizer isso, nao renderizar um cartao sem fala ao lado."""
    um_board(work)
    (work / "script.md").write_text(
        ROTEIRO.split("## beat 2")[0], encoding="utf-8")

    corpo = client.get("/v1/storyboard").text
    assert "nao existe mais no roteiro" in corpo


# --------------------------------------------------------------------------
# o destaque do ancora
# --------------------------------------------------------------------------


def test_ancora_fora_da_fronteira_de_palavra_nao_destaca_errado():
    """O mesmo erro que o `find_anchor` do storyboard tinha: "ostrar como"
    casa DENTRO de "mostrar como" e destacaria a partir do lugar errado."""
    markup = anchor_markup("vou mostrar como isso funciona", "ostrar como")
    assert "<mark>" not in markup
    assert "nao aparece literal" in markup


def test_ancora_ausente_e_informacao_de_revisao_e_nao_silencio():
    markup = anchor_markup("o custo esta no tempo", "o custo esta na camera")
    assert "similaridade" in markup


def test_ancora_com_pontuacao_ao_lado_ainda_destaca():
    markup = anchor_markup("O custo, na verdade, nao esta na camera.",
                           "nao esta na camera")
    assert "<mark>nao esta na camera</mark>" in markup


def test_o_destaque_nao_quebra_o_escape():
    """O texto e escapado nos tres trechos — antes, dentro e depois do
    destaque — entao `<b>` no roteiro nao vira tag de verdade."""
    markup = anchor_markup("ele disse <b>isso</b> mesmo", "disse <b>isso</b>")
    assert "<mark>disse &lt;b&gt;isso&lt;/b&gt;</mark>" in markup
    assert "<b>" not in markup


# --------------------------------------------------------------------------
# a mesma leitura na CLI e na UI
# --------------------------------------------------------------------------


@pytest.mark.parametrize("prepara,esperado", [
    (lambda work: None, progress.WRITTEN),
    (lambda work: approval.grant(
        work, "script", read_script(work / "script.md").digest()), progress.APPROVED),
    (lambda work: (
        approval.grant(work, "script", "digest-de-outra-versao")), progress.EDITED),
])
def test_cli_e_ui_dizem_a_mesma_coisa(client, work, cfg, capsys, prepara, esperado):
    """Duplicar esta leitura seria a divergencia que nao falha: a UI dizendo
    "aprovado" e o `status` dizendo "editado!", e quem olhasse a tela erraria
    com confianca. Os dois passam por `progress`, e este teste compara as duas
    saidas de verdade em cada um dos tres estados."""
    from pipeline.cli import cmd_status

    prepara(work)

    assert cmd_status(None, cfg) == 0
    da_cli = capsys.readouterr().out
    da_ui = client.get("/").text

    assert esperado in da_cli
    assert esperado in da_ui


def test_approval_corrompido_vale_como_nao_aprovado(client, work):
    approval.grant(work, "script", read_script(work / "script.md").digest())
    approval.path_for(work, "script").write_text("{", encoding="utf-8")

    assert progress.script_gate(work).state == progress.WRITTEN
    assert "Aprovar" in client.get("/v1/script").text


def test_json_do_approval_e_legivel_a_mao(client, work):
    """Ele e um artefato do pipeline como os outros: alguem vai abrir."""
    client.post("/v1/script/approve")
    dados = json.loads(approval.path_for(work, "script").read_text(encoding="utf-8"))
    assert set(dados) == {"digest", "approved_at"}
