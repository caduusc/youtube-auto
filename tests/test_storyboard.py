"""Validacao do storyboard.

A regra que domina o desenho e negativa: o storyboard NAO escreve segundo
absoluto. Ele escreve um trecho literal do roteiro, e o alinhamento descobre
depois onde aquele trecho caiu na fala real — e e isso que faz improviso na
gravacao nao desalinhar as imagens.
"""

from __future__ import annotations

import pytest

from pipeline.config import EditorialConfig, RenderConfig
from pipeline.schemas import PlannedBeat, Script, ScriptBeat
from pipeline.storyboard import find_anchor, normalize, render_text, resolve, validate


@pytest.fixture
def script() -> Script:
    return Script(
        slug="x", subject="assunto", argument="argumento",
        viewer_takeaway="lição", duration_target_seconds=120.0,
        beats=[
            ScriptBeat(id=1, title="abertura",
                       text="Eu gravei esse vídeo pra mostrar como monto a esteira."),
            ScriptBeat(id=2, title="o custo",
                       text="O custo todo cabe em duzentos reais por mês, "
                            "e isso inclui as imagens geradas."),
        ],
    )


@pytest.fixture
def rules() -> EditorialConfig:
    return EditorialConfig(
        broll_min_seconds=3.0, broll_max_seconds=9.0, max_switches_per_minute=10,
        intro_aroll_seconds=8.0, broll_ratio_min=0.35, broll_ratio_max=0.70,
    )


@pytest.fixture
def render() -> RenderConfig:
    return RenderConfig(max_sub_shots=4, sub_shot_seconds=2.5)


def beat(**kw) -> PlannedBeat:
    base = dict(
        beat_id=1, script_anchor="mostrar como monto a esteira",
        concept="a wooden desk with a microphone", concept_tags=["desk", "microphone"],
        sub_shots=2, rationale="porque a abertura precisa do lugar",
    )
    return PlannedBeat(**{**base, **kw})


# --------------------------------------------------------------------------
# o ancora
# --------------------------------------------------------------------------


def test_ancora_literal_e_aceito(script, rules, render):
    # 2 sub-planos de 2.5s = 5s de tela; com 10s falados, entre 3.5s e 7s
    assert validate([beat()], script, rules, render, spoken_seconds=10.0) == []


@pytest.mark.parametrize("variante", [
    "mostrar como monto a esteira",
    "Mostrar como monto a esteira",     # maiuscula
    "mostrar  como   monto a esteira",  # espaco duplo
    "mostrar como monto a esteira,",    # pontuacao sobrando
    "mostrar como monto a esteíra",     # acento diferente
])
def test_ancora_tolera_diferenca_que_nao_muda_o_lugar(script, variante):
    """O modelo copia o trecho, mas copia com a virgula de fora.

    Nenhuma dessas diferencas muda ONDE a imagem entra, e rejeitar por causa
    delas gastaria tentativa de API num problema que nao e editorial.
    """
    assert find_anchor(variante, script.beat(1).text) >= 0


def test_ancora_inventado_e_rejeitado_com_o_texto(script, rules, render):
    problemas = validate(
        [beat(script_anchor="uma frase que nao esta no roteiro")],
        script, rules, render, spoken_seconds=20.0,
    )
    assert any("nao aparece no texto" in p for p in problemas)
    assert any("uma frase que nao esta" in p for p in problemas)


def test_ancora_do_beat_errado_e_rejeitado(script, rules, render):
    """O texto existe no roteiro, mas em OUTRO beat."""
    problemas = validate(
        [beat(beat_id=1, script_anchor="duzentos reais por mês")],
        script, rules, render, spoken_seconds=20.0,
    )
    assert any("nao aparece no texto" in p for p in problemas)


def test_posicao_do_ancora_e_em_palavras(script):
    texto = script.beat(1).text
    assert find_anchor("Eu gravei", texto) == 0
    assert find_anchor("mostrar como", texto) == 5


def test_ancora_vazio_nao_casa_com_tudo(script):
    """String vazia acha posicao 0 em qualquer texto — nao pode passar."""
    assert find_anchor("", script.beat(1).text) == -1
    assert find_anchor("   ", script.beat(1).text) == -1


# --------------------------------------------------------------------------
# beat inexistente
# --------------------------------------------------------------------------


def test_beat_que_nao_existe_lista_os_validos(script, rules, render):
    problemas = validate([beat(beat_id=9)], script, rules, render, spoken_seconds=20.0)
    assert any("beat 9 nao existe" in p and "[1, 2]" in p for p in problemas)


# --------------------------------------------------------------------------
# sub-planos: o limite e geometrico
# --------------------------------------------------------------------------


def test_sub_shots_acima_do_teto_explica_a_geometria(script, rules, render):
    problemas = validate([beat(sub_shots=6)], script, rules, render, spoken_seconds=20.0)
    assert any("limite e 1 a 4" in p and "geometrico" in p for p in problemas)


def test_sub_shots_zero_e_rejeitado(script, rules, render):
    problemas = validate([beat(sub_shots=0)], script, rules, render, spoken_seconds=20.0)
    assert any("sub-planos" in p for p in problemas)


def test_teto_de_sub_shots_vem_do_config(script, rules):
    apertado = RenderConfig(max_sub_shots=2, sub_shot_seconds=2.5)
    assert validate([beat(sub_shots=2)], script, rules, apertado, spoken_seconds=10.0) == []
    assert validate([beat(sub_shots=3)], script, rules, apertado, spoken_seconds=10.0)


# --------------------------------------------------------------------------
# tags e rationale
# --------------------------------------------------------------------------


def test_tags_fora_da_faixa(script, rules, render):
    problemas = validate([beat(concept_tags=["so_uma"])], script, rules, render, spoken_seconds=20.0)
    assert any("concept_tags" in p for p in problemas)


def test_rationale_vazio_e_rejeitado(script, rules, render):
    """E o que se le na revisao para decidir se aprova."""
    problemas = validate([beat(rationale="  ")], script, rules, render, spoken_seconds=20.0)
    assert any("rationale" in p for p in problemas)


# --------------------------------------------------------------------------
# duas imagens no mesmo beat do roteiro
# --------------------------------------------------------------------------


def test_duas_imagens_no_mesmo_beat_em_ordem_e_valido(script, rules, render):
    """Um beat longo pode pedir duas imagens."""
    problemas = validate(
        [
            beat(beat_id=2, script_anchor="O custo todo", sub_shots=1),
            beat(beat_id=2, script_anchor="as imagens geradas", sub_shots=1),
        ],
        script, rules, render, spoken_seconds=10.0,
    )
    assert problemas == []


def test_duas_imagens_fora_de_ordem_sao_rejeitadas(script, rules, render):
    """Duas imagens no mesmo ponto da fala disputam a tela."""
    problemas = validate(
        [
            beat(beat_id=2, script_anchor="as imagens geradas", sub_shots=1),
            beat(beat_id=2, script_anchor="O custo todo", sub_shots=1),
        ],
        script, rules, render, spoken_seconds=10.0,
    )
    assert any("ordem crescente" in p for p in problemas)


def test_duas_imagens_na_mesma_ancora_sao_rejeitadas(script, rules, render):
    problemas = validate(
        [beat(beat_id=2, script_anchor="O custo todo", sub_shots=1),
         beat(beat_id=2, script_anchor="O custo todo", sub_shots=1)],
        script, rules, render, spoken_seconds=10.0,
    )
    assert any("ordem crescente" in p for p in problemas)


# --------------------------------------------------------------------------
# cobertura
# --------------------------------------------------------------------------


def test_cobertura_abaixo_do_piso_diz_quanto_falta(script, rules, render):
    # 1 sub-plano de 2.5s em 100s falados = 2.5%, piso e 35%
    problemas = validate([beat(sub_shots=1)], script, rules, render, spoken_seconds=100.0)
    assert any("abaixo do minimo" in p and "faltam" in p for p in problemas)


def test_cobertura_acima_do_teto(script, rules, render):
    problemas = validate(
        [beat(sub_shots=4), beat(beat_id=2, script_anchor="O custo todo", sub_shots=4)],
        script, rules, render, spoken_seconds=20.0,
    )
    assert any("acima do maximo" in p for p in problemas)


def test_cobertura_na_mosca_nao_e_rejeitada(script, rules, render):
    """O bug que custou uma rodada no estagio 3 antigo: sem tolerancia, uma
    proposta que acerta a proporcao exata cai em 0.4999 e e rejeitada com uma
    mensagem que o modelo nao tem como corrigir."""
    # 14 sub-planos de 2.5s = 35.0s; piso = 100 * 0.35 = 35.0s exatos
    itens = [beat(sub_shots=4), beat(sub_shots=4),
             beat(sub_shots=4), beat(sub_shots=2)]
    # ancoras distintas e em ordem para nao cair na outra regra
    textos = ["Eu gravei", "esse vídeo", "pra mostrar", "como monto"]
    itens = [beat(sub_shots=n, script_anchor=t)
             for n, t in zip([4, 4, 4, 2], textos)]
    problemas = validate(itens, script, rules, render, spoken_seconds=100.0)
    assert not any("abaixo do minimo" in p for p in problemas), problemas


def test_sem_duracao_estimada_nao_checa_cobertura(script, rules, render):
    assert validate([beat(sub_shots=1)], script, rules, render, spoken_seconds=0.0) == []


def test_sub_shots_invalido_nao_conta_na_cobertura(script, rules, render):
    """Senao um sub_shots=99 satisfaria a cobertura e o erro real ficaria
    escondido atras de 'cobertura ok'."""
    problemas = validate([beat(sub_shots=99)], script, rules, render, spoken_seconds=100.0)
    assert any("limite e 1 a 4" in p for p in problemas)
    assert any("abaixo do minimo" in p for p in problemas)


def test_storyboard_vazio(script, rules, render):
    assert validate([], script, rules, render, spoken_seconds=100.0) == ["o storyboard nao tem beat nenhum"]


# --------------------------------------------------------------------------
# resolve
# --------------------------------------------------------------------------


def test_resolve_grava_a_posicao_e_a_duracao(script, render):
    sb = resolve([beat()], script, render, input_hash="h")
    assert sb.input_hash == "h"
    assert sb.beats[0].anchor_offset == 5
    assert sb.beats[0].seconds_per_shot == 2.5
    assert sb.beats[0].screen_seconds == 5.0
    assert sb.n_images == 1


def test_resolve_ordena_por_beat_e_posicao(script, render):
    sb = resolve(
        [beat(beat_id=2, script_anchor="as imagens geradas"),
         beat(beat_id=1, script_anchor="mostrar como"),
         beat(beat_id=2, script_anchor="O custo todo")],
        script, render, input_hash="h",
    )
    assert [(b.beat_id, b.anchor_offset) for b in sb.beats] == \
        sorted((b.beat_id, b.anchor_offset) for b in sb.beats)


def test_render_text_mostra_ancora_e_porque(script, render):
    """Sem o trecho do roteiro ao lado, voce aprova uma lista de descricoes de
    imagem sem saber sobre o que cada uma entra."""
    saida = render_text(resolve([beat()], script, render, input_hash="h"), script)
    assert "beat 1 — abertura" in saida
    assert "mostrar como monto a esteira" in saida
    assert "porque a abertura precisa do lugar" in saida
    assert "1 imagem" in saida


def test_normalize():
    assert normalize("Ação, é só!") == "acao  e so"


def test_ancora_nao_casa_no_meio_de_palavra(script):
    """Busca por sequencia de palavras, nao por substring.

    "ostrar como" casaria dentro de "mostrar como" e devolveria posicao 6 em
    vez de -1 — passaria pela validacao e alinharia a imagem no lugar errado.
    """
    texto = script.beat(1).text
    assert find_anchor("mostrar como", texto) == 5
    assert find_anchor("ostrar como", texto) == -1
    assert find_anchor("esteir", texto) == -1


def test_ancora_de_uma_palavra_so_funciona(script):
    assert find_anchor("esteira", script.beat(1).text) == 9   # eu=0 ... a=8, esteira=9
