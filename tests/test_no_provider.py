"""Rodar sem conta de geracao de imagem.

Caso real: o usuario tem credito na API da Anthropic (estagio 3) e chave do
Pexels (gratuita), mas nao tem credito no Replicate. O pipeline precisa
produzir video nessa configuracao, nao morrer.
"""

from __future__ import annotations

import pytest
import requests
from conftest import span
from test_bank_reuse import FakeStock, bank  # noqa: F401

from pipeline.resolver import AssetResolver

from pipeline.edl import resolve as resolve_edl
from pipeline.providers import ReplicateProvider, build_provider
from pipeline.schemas import PlannedEDL
from pipeline.stages.assets import build_resolver


def resolver_without_provider(bank, tmp_path, stock):
    """Sem provider DE VERDADE.

    O `make_resolver` de test_bank_reuse troca provider=None por um
    FakeProvider, entao ele nao serve para este caso.
    """
    return AssetResolver(
        bank=bank, provider=None, stock=stock,
        style_suffix="flat editorial illustration",
        budget_usd=1.50, cost_usd_per_image=0.0,
        images_dir=tmp_path / "assets" / "images",
    )


def two_brolls(transcript):
    generic = span("broll", 6, 8)
    generic.concept, generic.concept_tags = "a city skyline at dusk", ["city", "skyline"]
    specific = span("broll", 12, 14)
    specific.concept, specific.concept_tags = "a chessboard mid-game", ["chessboard", "chess"]
    spans = [span("aroll", 0, 5), generic, span("aroll", 9, 11), specific,
             span("aroll", 15, 199)]
    return resolve_edl(PlannedEDL(spans=spans), transcript)


# --------------------------------------------------------------------------
# active: none
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["none", "off", "disabled"])
def test_build_provider_aceita_desligar(value, config):
    assert build_provider(value, config.image_provider) is None


def test_active_invalido_ainda_e_erro(config):
    with pytest.raises(RuntimeError, match="nao tem implementacao"):
        build_provider("dall-e", config.image_provider)


def test_mensagem_de_active_invalido_cita_none(config):
    with pytest.raises(RuntimeError, match="'replicate' ou 'none'"):
        build_provider("dall-e", config.image_provider)


def test_estagio_constroi_resolver_sem_provider(config, bank, tmp_path, monkeypatch):
    """Sem isso o estagio morria antes de chegar ao resolver, que sempre
    soube lidar com provider=None."""
    monkeypatch.setenv("PEXELS_API_KEY", "fake")
    cfg = config.model_copy(deep=True)
    cfg.root = tmp_path
    cfg.image_provider.active = "none"

    resolver = build_resolver(cfg, bank, dry_run=False)
    assert resolver.provider is None
    assert resolver.stock is not None          # stock continua valendo


def test_stock_resolve_e_o_resto_vira_cor_solida(bank, tmp_path, transcript):
    """Com geracao desligada: o generico sai do stock, o especifico vira cor
    solida, e o video sai."""
    stock = FakeStock(["city"])
    assets = resolver_without_provider(bank, tmp_path, stock).resolve(two_brolls(transcript))

    assert assets.by_origin == {"stock": 1, "solid": 1}
    assert assets.total_cost_usd == 0.0
    solid = next(i for i in assets.items if i.origin == "solid")
    assert "sem provider" in solid.note


def test_estimativa_sem_provider_nao_promete_geracao(bank, tmp_path, transcript):
    stock = FakeStock(["city"])
    assets = resolver_without_provider(bank, tmp_path, stock).resolve(
        two_brolls(transcript), dry_run=True)

    assert assets.estimate.worst_case_usd == 0.0
    assert assets.estimate.within_budget is True


# --------------------------------------------------------------------------
# erro de conta no provider
# --------------------------------------------------------------------------


def make_http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} error", response=response)


def test_sem_credito_explica_e_aponta_o_escape():
    explained = ReplicateProvider._explain_http_error(make_http_error(402))
    assert "sem credito" in explained
    assert "image_provider.active: none" in explained


@pytest.mark.parametrize("status", [401, 403])
def test_credencial_recusada_aponta_a_variavel(status):
    explained = ReplicateProvider._explain_http_error(make_http_error(status))
    assert "REPLICATE_API_TOKEN" in explained


@pytest.mark.parametrize("status", [500, 502, 429])
def test_erro_transitorio_nao_recebe_explicacao_de_conta(status):
    """5xx e 429 sao retentaveis; nao podem virar mensagem de configuracao."""
    assert ReplicateProvider._explain_http_error(make_http_error(status)) is None
