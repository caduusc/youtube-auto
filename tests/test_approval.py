"""Portoes de aprovacao.

O caso que domina o desenho: a aprovacao guarda O QUE foi aprovado, nao apenas
que houve aprovacao. Sem isso existe um jeito silencioso de errar — aprovar o
roteiro, editar o texto, e o estagio seguinte rodar no texto novo carregando o
aval do antigo, gastando uma chamada de Opus e gerando imagens para uma versao
que nao existe mais.
"""

from __future__ import annotations

import json

import pytest

from pipeline import approval
from pipeline.approval import NotApproved


def test_sem_aprovacao_o_erro_diz_o_comando(tmp_path):
    """Quem le a mensagem e uma pessoa: 'nao aprovado' manda procurar."""
    with pytest.raises(NotApproved, match="pipeline approve script x"):
        approval.require(tmp_path, "script", "abc",
                         what="O roteiro", command="pipeline approve script x")


def test_aprovado_com_o_mesmo_digest_passa(tmp_path):
    approval.grant(tmp_path, "script", "abc")
    approval.require(tmp_path, "script", "abc", what="O roteiro", command="x")


def test_editar_depois_de_aprovar_revoga(tmp_path):
    """O caso que motiva guardar o digest."""
    approval.grant(tmp_path, "script", "digest-da-v1")
    with pytest.raises(NotApproved, match="mudou depois de aprovado"):
        approval.require(tmp_path, "script", "digest-da-v2",
                         what="O roteiro", command="pipeline approve script x")


def test_a_mensagem_de_revogacao_diz_quando_foi_aprovado(tmp_path):
    aprovacao = approval.grant(tmp_path, "script", "v1")
    with pytest.raises(NotApproved, match=aprovacao.approved_at):
        approval.require(tmp_path, "script", "v2", what="O roteiro", command="x")


def test_portoes_sao_independentes(tmp_path):
    approval.grant(tmp_path, "script", "abc")
    with pytest.raises(NotApproved):
        approval.require(tmp_path, "storyboard", "abc", what="O storyboard", command="x")


def test_revoke(tmp_path):
    approval.grant(tmp_path, "script", "abc")
    assert approval.revoke(tmp_path, "script") is True
    assert approval.read(tmp_path, "script") is None
    assert approval.revoke(tmp_path, "script") is False   # idempotente


def test_arquivo_corrompido_vale_como_ausencia(tmp_path):
    """O caminho seguro e pedir de novo, nao adivinhar que estava aprovado."""
    approval.path_for(tmp_path, "script").write_text("{ nao e json", encoding="utf-8")
    assert approval.read(tmp_path, "script") is None
    with pytest.raises(NotApproved):
        approval.require(tmp_path, "script", "abc", what="O roteiro", command="x")


def test_arquivo_sem_o_campo_digest_vale_como_ausencia(tmp_path):
    approval.path_for(tmp_path, "script").write_text(
        json.dumps({"approved_at": "2026-01-01T00:00:00Z"}), encoding="utf-8")
    assert approval.read(tmp_path, "script") is None


def test_grant_cria_o_diretorio(tmp_path):
    fundo = tmp_path / "a" / "b" / "c"
    approval.grant(fundo, "script", "abc")
    assert approval.path_for(fundo, "script").exists()


def test_o_arquivo_e_legivel_por_gente(tmp_path):
    """Voce vai olhar esse arquivo quando algo nao fizer sentido."""
    approval.grant(tmp_path, "script", "abcdef123456")
    dados = json.loads(approval.path_for(tmp_path, "script").read_text(encoding="utf-8"))
    assert dados["digest"] == "abcdef123456"
    assert dados["approved_at"].endswith("Z")
