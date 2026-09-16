"""Rodar sem conta de geracao de imagem.

Caso real: o usuario tem credito na API da Anthropic (estagio 3) e chave do
Pexels (gratuita), mas nao tem credito no Replicate. O pipeline precisa
produzir video nessa configuracao, nao morrer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests
from conftest import span
from test_bank_reuse import FakeStock, bank  # noqa: F401

from pipeline.resolver import AssetResolver, from_edl

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
    assert build_provider(value, config.image_provider, config.network) is None


def test_active_invalido_ainda_e_erro(config):
    with pytest.raises(RuntimeError, match="nao tem implementacao"):
        build_provider("dall-e", config.image_provider, config.network)


def test_mensagem_de_active_invalido_cita_none(config):
    with pytest.raises(RuntimeError, match="'replicate' ou 'none'"):
        build_provider("dall-e", config.image_provider, config.network)


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
    assets = resolver_without_provider(bank, tmp_path, stock).resolve(from_edl(two_brolls(transcript)))

    assert assets.by_origin == {"stock": 1, "solid": 1}
    assert assets.total_cost_usd == 0.0
    solid = next(i for i in assets.items if i.origin == "solid")
    assert "sem provider" in solid.note


def test_estimativa_sem_provider_nao_promete_geracao(bank, tmp_path, transcript):
    stock = FakeStock(["city"])
    assets = resolver_without_provider(bank, tmp_path, stock).resolve(
        from_edl(two_brolls(transcript)), dry_run=True)

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


# --------------------------------------------------------------------------
# o input do provider vem do config
# --------------------------------------------------------------------------


def test_o_provider_manda_o_input_do_config(config, monkeypatch):
    """Era um dict fixo no codigo com `num_outputs: 1` dentro, o que tornava
    "provider plugavel" falso: o flux-1.1-pro recusa `num_outputs`."""
    cfg = config.model_copy(deep=True)
    cfg.image_provider.replicate.input = {
        "aspect_ratio": "1:1", "guidance": 4.5, "megapixels": "1"}
    monkeypatch.setenv(cfg.image_provider.replicate.env, "token-de-teste")

    enviados = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self): ...

        def json(self):
            return {"status": "succeeded", "output": ["https://x/y.png"]}

    def fake_post(url, **kwargs):
        enviados.update(kwargs["json"]["input"])
        return FakeResponse()

    monkeypatch.setattr("pipeline.providers.requests.post", fake_post)
    provider = ReplicateProvider(cfg.image_provider.replicate, cfg.network)
    monkeypatch.setattr("pipeline.providers._download", lambda *a, **k: None)
    provider.generate("um banco de madeira", Path("/tmp/x.png"))

    assert enviados["aspect_ratio"] == "1:1"
    assert enviados["guidance"] == 4.5
    assert "num_outputs" not in enviados, "mandou uma chave que o config nao pediu"
    assert enviados["prompt"] == "um banco de madeira"
    assert enviados["output_format"] == cfg.image_provider.replicate.output_format


def test_o_config_nao_pode_sobrescrever_o_prompt(config, monkeypatch):
    """O prompt e do storyboard. Uma chave `prompt` no YAML nao pode
    substituir o que o modelo escreveu."""
    cfg = config.model_copy(deep=True)
    cfg.image_provider.replicate.input = {"prompt": "um gato de chapeu"}
    monkeypatch.setenv(cfg.image_provider.replicate.env, "token-de-teste")

    enviados = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self): ...

        def json(self):
            return {"status": "succeeded", "output": ["https://x/y.png"]}

    monkeypatch.setattr(
        "pipeline.providers.requests.post",
        lambda url, **kw: (enviados.update(kw["json"]["input"]), FakeResponse())[1])
    monkeypatch.setattr("pipeline.providers._download", lambda *a, **k: None)
    ReplicateProvider(cfg.image_provider.replicate, cfg.network).generate(
        "um banco de madeira", Path("/tmp/x.png"))

    assert enviados["prompt"] == "um banco de madeira"
