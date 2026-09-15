"""Invalidacao de cache dos estagios 3 e 4.

Cada estagio e cacheado pelo hash da entrada. A questao e o que conta como
"entrada": se so o artefato anterior, mexer no config nao regera nada — e o
fluxo de calibragem consiste justamente em mexer no config.
"""

from __future__ import annotations

from conftest import alternating

from pipeline.edl import build
from pipeline.stages.assets import cache_key as assets_key
from pipeline.stages.plan import cache_key as plan_key


def edl_of(transcript):
    return build(alternating(aroll_len=4, broll_len=5), transcript,
                 model="claude-opus-5", attempts=1)


# --------------------------------------------------------------------------
# estagio 3
# --------------------------------------------------------------------------


def test_mesmo_transcript_e_mesmas_regras_reusa(transcript, config):
    assert plan_key(transcript, config) == plan_key(transcript, config)


def test_afrouxar_as_regras_invalida_a_edl(transcript, config):
    """Baixar `broll_ratio_min` e pedir uma EDL diferente. Reusar a antiga
    esconde a mudanca e o usuario fica sem entender por que nada mudou."""
    other = config.model_copy(deep=True)
    other.editorial.broll_ratio_min = 0.35
    assert plan_key(transcript, config) != plan_key(transcript, other)


def test_mudar_a_intro_invalida_a_edl(transcript, config):
    other = config.model_copy(deep=True)
    other.editorial.intro_aroll_seconds = 8.0
    assert plan_key(transcript, config) != plan_key(transcript, other)


def test_mudar_o_ritmo_invalida_a_edl(transcript, config):
    other = config.model_copy(deep=True)
    other.editorial.max_switches_per_minute = 4
    assert plan_key(transcript, config) != plan_key(transcript, other)


def test_mudar_config_irrelevante_nao_invalida_a_edl(transcript, config):
    """O estagio 3 nao usa o estilo das imagens; mexer nele nao pode custar
    outra chamada a API."""
    other = config.model_copy(deep=True)
    other.style_suffix = "totalmente outra estetica"
    other.render.crf = 28
    assert plan_key(transcript, config) == plan_key(transcript, other)


# --------------------------------------------------------------------------
# estagio 4
# --------------------------------------------------------------------------


def test_mesma_edl_e_mesmo_estilo_reusa(transcript, config):
    edl = edl_of(transcript)
    assert assets_key(edl, config) == assets_key(edl, config)


def test_mudar_o_style_suffix_invalida_os_assets(transcript, config):
    """E o campo que o README manda calibrar. Sem ele na chave, calibrar
    nao faz efeito nenhum."""
    edl = edl_of(transcript)
    other = config.model_copy(deep=True)
    other.style_suffix = "watercolor on cold-press paper, loose washes"
    assert assets_key(edl, config) != assets_key(edl, other)


def test_mudar_o_limiar_de_similaridade_invalida_os_assets(transcript, config):
    """Muda a rota de cada segmento entre banco e geracao."""
    edl = edl_of(transcript)
    other = config.model_copy(deep=True)
    other.bank.similarity_threshold = 0.90
    assert assets_key(edl, config) != assets_key(edl, other)


def test_mudar_as_tags_genericas_invalida_os_assets(transcript, config):
    edl = edl_of(transcript)
    other = config.model_copy(deep=True)
    other.stock.generic_tags = [*config.stock.generic_tags, "server rack"]
    assert assets_key(edl, config) != assets_key(edl, other)


def test_ordem_das_tags_nao_importa(transcript, config):
    """Reordenar a lista no YAML nao e uma mudanca de comportamento."""
    edl = edl_of(transcript)
    other = config.model_copy(deep=True)
    other.stock.generic_tags = list(reversed(config.stock.generic_tags))
    assert assets_key(edl, config) == assets_key(edl, other)


def test_mudar_o_modelo_do_provider_invalida_os_assets(transcript, config):
    edl = edl_of(transcript)
    other = config.model_copy(deep=True)
    other.image_provider.replicate.model = "black-forest-labs/flux-dev"
    assert assets_key(edl, config) != assets_key(edl, other)


def test_edl_diferente_invalida_os_assets(transcript, config):
    a = edl_of(transcript)
    b = build(alternating(aroll_len=5, broll_len=5), transcript,
              model="claude-opus-5", attempts=1)
    assert assets_key(a, config) != assets_key(b, config)


def test_ligar_o_briefing_invalida_a_edl(transcript, config):
    """O briefing muda os concepts, entao mexer nele tem que refazer a EDL."""
    other = config.model_copy(deep=True)
    other.anthropic.two_phase = not config.anthropic.two_phase
    assert plan_key(transcript, config) != plan_key(transcript, other)
