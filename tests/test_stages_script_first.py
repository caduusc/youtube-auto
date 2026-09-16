"""Os estagios de roteiro-primeiro, com os portoes.

A API e mockada. O que estes testes verificam e o que o pipeline faz com o
DISCO: o que ele reusa, o que ele reescreve, e o que ele se recusa a fazer sem
aprovacao — porque cada uma dessas decisoes ou protege uma edicao sua ou evita
uma chamada de Opus jogada fora.
"""

from __future__ import annotations

import pytest

from pipeline import approval, stages
from pipeline.approval import NotApproved
from pipeline.schemas import (
    PlannedBeat, PlannedScript, PlannedStoryboard, ScriptBeat,
)
from pipeline.script import read as read_script
from pipeline.stages.storyboard import cache_key
from test_agents import FakeClient, FakeResponse


@pytest.fixture
def cfg(config, tmp_path):
    """O config de exemplo, com work/ isolado no tmp."""
    return config.model_copy(update={"root": tmp_path})


def um_roteiro(n=2, palavras=30) -> PlannedScript:
    return PlannedScript(
        subject="assunto do video", argument="o argumento central",
        viewer_takeaway="o que o espectador leva",
        beats=[ScriptBeat(id=i, title=f"t{i}", text=" ".join(["palavra"] * palavras))
               for i in range(1, n + 1)],
    )


def um_board(sub_shots=3) -> PlannedStoryboard:
    """Cobertura dentro da faixa do `rules` da suite (50-70%).

    Dois beats de 3 sub-planos de 2.5s = 15s, contra 24s falados = 62%.
    """
    return PlannedStoryboard(beats=[
        PlannedBeat(
            beat_id=i, script_anchor="palavra palavra palavra",
            concept=f"a wooden desk with a ledger and a lamp, view {i}",
            concept_tags=["desk", "ledger"], sub_shots=sub_shots,
            rationale="porque o argumento precisa de imagem",
        )
        for i in (1, 2)
    ])


# --------------------------------------------------------------------------
# estagio 1
# --------------------------------------------------------------------------


def test_escreve_o_arquivo(cfg):
    client = FakeClient([FakeResponse(um_roteiro())])
    script = stages.script.run("minha ideia", slug="v1", target_seconds=60.0,
                               config=cfg, client=client)
    caminho = cfg.work_dir / "v1" / "script.md"
    assert caminho.exists()
    assert read_script(caminho) == script


def test_nao_reescreve_por_cima_da_sua_edicao(cfg):
    """O cache e por ARQUIVO, nao por hash da ideia.

    Depois de o roteiro existir a fonte da verdade e o arquivo que voce editou.
    Reescrever por cima apagaria a edicao — e nem uma chamada de API seria
    gasta para fazer isso, o que e pior: seria silencioso.
    """
    client = FakeClient([FakeResponse(um_roteiro())])
    stages.script.run("ideia", slug="v1", target_seconds=60.0, config=cfg, client=client)

    caminho = cfg.work_dir / "v1" / "script.md"
    editado = caminho.read_text(encoding="utf-8").replace("palavra", "EDITADO", 1)
    caminho.write_text(editado, encoding="utf-8")

    # segunda chamada: nenhuma resposta de API disponivel, e nao precisa
    segundo = stages.script.run("ideia", slug="v1", target_seconds=60.0,
                                config=cfg, client=FakeClient([]))
    assert "EDITADO" in segundo.beats[0].text


def test_slug_sai_da_ideia():
    """O roteiro nasce antes do arquivo de video, entao nao ha nome de onde
    tirar o slug como o `ingest` faz."""
    assert stages.script.slug_for("Como montar um estúdio de vídeo em casa hoje") \
        == "como-montar-um-estudio-de-video"


# --------------------------------------------------------------------------
# o portao entre 1 e 2
# --------------------------------------------------------------------------


def test_storyboard_recusa_sem_aprovacao(cfg):
    """Rodar o storyboard num roteiro que voce ainda vai reescrever joga uma
    chamada de Opus fora."""
    client = FakeClient([FakeResponse(um_roteiro())])
    script = stages.script.run("ideia", slug="v1", target_seconds=60.0,
                               config=cfg, client=client)

    with pytest.raises(NotApproved, match="pipeline approve script v1"):
        stages.storyboard.run(script, config=cfg, client=FakeClient([]))


def test_storyboard_roda_depois_de_aprovar(cfg):
    client = FakeClient([FakeResponse(um_roteiro())])
    script = stages.script.run("ideia", slug="v1", target_seconds=60.0,
                               config=cfg, client=client)
    work = cfg.work_dir / "v1"
    approval.grant(work, "script", script.digest())

    board = stages.storyboard.run(script, config=cfg,
                                  client=FakeClient([FakeResponse(um_board())]))
    assert board.n_images == 2
    assert (work / "storyboard.json").exists()


def test_editar_o_roteiro_depois_de_aprovar_bloqueia_o_storyboard(cfg):
    """O caso silencioso: sem guardar o digest, o storyboard rodaria no texto
    novo com o aval do antigo."""
    client = FakeClient([FakeResponse(um_roteiro())])
    script = stages.script.run("ideia", slug="v1", target_seconds=60.0,
                               config=cfg, client=client)
    work = cfg.work_dir / "v1"
    approval.grant(work, "script", script.digest())

    caminho = work / "script.md"
    caminho.write_text(
        caminho.read_text(encoding="utf-8").replace("palavra", "OUTRA COISA", 1),
        encoding="utf-8")
    editado = read_script(caminho)

    with pytest.raises(NotApproved, match="mudou depois de aprovado"):
        stages.storyboard.run(editado, config=cfg, client=FakeClient([]))


# --------------------------------------------------------------------------
# cache do estagio 2
# --------------------------------------------------------------------------


def test_storyboard_e_cacheado(cfg):
    client = FakeClient([FakeResponse(um_roteiro())])
    script = stages.script.run("ideia", slug="v1", target_seconds=60.0,
                               config=cfg, client=client)
    approval.grant(cfg.work_dir / "v1", "script", script.digest())

    stages.storyboard.run(script, config=cfg, client=FakeClient([FakeResponse(um_board())]))
    # segunda chamada nao tem resposta disponivel: vem do cache
    board = stages.storyboard.run(script, config=cfg, client=FakeClient([]))
    assert board.n_images == 2


@pytest.mark.parametrize("mudanca", [
    {"render": "sub_shot_seconds"},
    {"render": "max_sub_shots"},
    {"script": "words_per_minute"},
])
def test_config_que_muda_a_decisao_invalida_o_cache(cfg, mudanca):
    """Baixar `sub_shot_seconds` nao pode deixar de pe um storyboard calculado
    para a duracao antiga."""
    client = FakeClient([FakeResponse(um_roteiro())])
    script = stages.script.run("ideia", slug="v1", target_seconds=60.0,
                               config=cfg, client=client)

    antes = cache_key(script, cfg)
    (secao, campo), = mudanca.items()
    sub = getattr(cfg, secao)
    novo = cfg.model_copy(update={
        secao: sub.model_copy(update={campo: getattr(sub, campo) * 2})
    })
    assert cache_key(script, novo) != antes, f"{secao}.{campo} nao invalidou"


def test_editar_o_texto_do_beat_invalida_o_cache(cfg):
    """O `concept` daquele beat deveria mudar se o texto dele mudou."""
    client = FakeClient([FakeResponse(um_roteiro())])
    script = stages.script.run("ideia", slug="v1", target_seconds=60.0,
                               config=cfg, client=client)
    antes = cache_key(script, cfg)

    outro = script.model_copy(deep=True)
    outro.beats[0].text = "texto completamente diferente"
    assert cache_key(outro, cfg) != antes


def test_estilo_invalida_o_cache(cfg):
    """O `concept` e escrito para um estilo."""
    client = FakeClient([FakeResponse(um_roteiro())])
    script = stages.script.run("ideia", slug="v1", target_seconds=60.0,
                               config=cfg, client=client)
    antes = cache_key(script, cfg)
    outro = cfg.model_copy(update={"style": "", "style_suffix": "um estilo bem diferente"})
    assert cache_key(script, outro) != antes
