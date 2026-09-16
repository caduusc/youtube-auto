"""Os dois agentes novos: roteiro e storyboard.

A API e mockada — nenhum teste gasta nada. O que estes testes verificam nao e
a qualidade do texto (isso nao da para testar), e o CONTRATO: o que acontece
quando o modelo devolve algo invalido, e se o erro especifico volta como
feedback em vez de uma nova tentativa no escuro.
"""

from __future__ import annotations

import json

import pytest

from pipeline.config import AnthropicConfig, EditorialConfig, RenderConfig, ScriptConfig
from pipeline.schemas import (
    PlannedBeat, PlannedScript, PlannedStoryboard, Script, ScriptBeat,
)
from pipeline.storyboard import StoryboardFailed, plan, script_block
from pipeline.storyboard import system_prompt as storyboard_prompt
from pipeline.writer import WritingFailed, validate, write
from pipeline.writer import system_prompt as writer_prompt


class FakeUsage:
    input_tokens = 2000
    output_tokens = 1500
    cache_read_input_tokens = 0


class FakeResponse:
    def __init__(self, parsed, stop_reason="end_turn", stop_details=None):
        self.parsed_output = parsed
        self.stop_reason = stop_reason
        self.stop_details = stop_details
        self.usage = FakeUsage()


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)

    def with_options(self, **_):
        return self


@pytest.fixture
def anthropic_config() -> AnthropicConfig:
    return AnthropicConfig(model="claude-opus-5", env="X", max_attempts=3)


@pytest.fixture
def script_rules() -> ScriptConfig:
    return ScriptConfig(words_per_minute=150.0)


@pytest.fixture
def rules() -> EditorialConfig:
    return EditorialConfig(
        broll_min_seconds=3.0, broll_max_seconds=9.0, max_switches_per_minute=10,
        intro_aroll_seconds=8.0, broll_ratio_min=0.35, broll_ratio_max=0.70,
    )


@pytest.fixture
def render() -> RenderConfig:
    return RenderConfig(max_sub_shots=4, sub_shot_seconds=2.5)


# --------------------------------------------------------------------------
# roteirista
# --------------------------------------------------------------------------


def a_script(n_beats: int = 3, palavras: int = 40) -> PlannedScript:
    return PlannedScript(
        subject="como montar uma esteira de producao de video",
        argument="a automacao ja e barata o bastante para sustentar um canal",
        viewer_takeaway="da para montar o estudio inteiro em software",
        beats=[
            ScriptBeat(id=i, title=f"beat {i}", text=" ".join(["palavra"] * palavras))
            for i in range(1, n_beats + 1)
        ],
    )


def test_roteiro_valido_na_primeira(anthropic_config, script_rules):
    client = FakeClient([FakeResponse(a_script())])
    s = write("minha ideia", slug="x", target_seconds=120.0,
              config=anthropic_config, rules=script_rules, client=client)
    assert s.slug == "x"
    assert s.duration_target_seconds == 120.0
    assert [b.id for b in s.beats] == [1, 2, 3]
    assert len(client.messages.calls) == 1


def test_ids_com_buraco_voltam_como_feedback(anthropic_config, script_rules):
    ruim = a_script()
    ruim.beats[1].id = 7                     # 1, 7, 3
    client = FakeClient([FakeResponse(ruim), FakeResponse(a_script())])

    write("ideia", slug="x", target_seconds=120.0,
          config=anthropic_config, rules=script_rules, client=client)

    assert len(client.messages.calls) == 2
    feedback = client.messages.calls[1]["messages"][-1]["content"]
    assert "sem buraco" in feedback and "[1, 7, 3]" in feedback


def test_beat_vazio_voltam_como_feedback(anthropic_config, script_rules):
    ruim = a_script()
    ruim.beats[0].text = "   "
    client = FakeClient([FakeResponse(ruim), FakeResponse(a_script())])

    write("ideia", slug="x", target_seconds=120.0,
          config=anthropic_config, rules=script_rules, client=client)

    assert "beat 1 esta sem texto" in client.messages.calls[1]["messages"][-1]["content"]


def test_desiste_depois_de_max_attempts(anthropic_config, script_rules):
    ruim = a_script()
    ruim.beats[1].id = 7
    client = FakeClient([FakeResponse(ruim) for _ in range(3)])

    with pytest.raises(WritingFailed, match="3 tentativas"):
        write("ideia", slug="x", target_seconds=120.0,
              config=anthropic_config, rules=script_rules, client=client)
    assert len(client.messages.calls) == 3


def test_recusa_da_api_nao_vira_retry(anthropic_config, script_rules):
    """Recusa nao e erro de validacao: insistir nao muda o resultado."""
    class Detalhe:
        category = "cyber"

    client = FakeClient([FakeResponse(a_script(), stop_reason="refusal",
                                      stop_details=Detalhe())])
    with pytest.raises(WritingFailed, match="recusou"):
        write("ideia", slug="x", target_seconds=120.0,
              config=anthropic_config, rules=script_rules, client=client)
    assert len(client.messages.calls) == 1


def test_truncado_aponta_o_config(anthropic_config, script_rules):
    client = FakeClient([FakeResponse(a_script(), stop_reason="max_tokens")])
    with pytest.raises(WritingFailed, match="anthropic.max_tokens"):
        write("ideia", slug="x", target_seconds=120.0,
              config=anthropic_config, rules=script_rules, client=client)


def test_a_ideia_dele_entra_no_pedido(anthropic_config, script_rules):
    client = FakeClient([FakeResponse(a_script())])
    write("quero falar sobre xadrez e automacao", slug="x", target_seconds=120.0,
          config=anthropic_config, rules=script_rules, client=client)
    pedido = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "xadrez e automacao" in pedido


def test_prompt_traduz_duracao_em_palavras(script_rules):
    """900s a 150 ppm = 2250 palavras, com folga de 15% para os dois lados."""
    p = writer_prompt(900.0, script_rules)
    assert "2250" in p
    assert "1912" in p and "2588" in p    # 0.85x e 1.15x


def test_prompt_nao_pede_indicacao_de_cena(script_rules):
    """Misturar roteiro e direcao de imagem na mesma resposta piora os dois."""
    p = writer_prompt(120.0, script_rules)
    assert "Nao escreva indicacao de cena" in p


def test_validate_roteiro_sem_beat():
    vazio = PlannedScript(subject="a", argument="b", viewer_takeaway="c", beats=[])
    assert validate(vazio) == ["o roteiro nao tem beat nenhum"]


# --------------------------------------------------------------------------
# storyboard
# --------------------------------------------------------------------------


@pytest.fixture
def script() -> Script:
    return Script(
        slug="x", subject="assunto", argument="argumento", viewer_takeaway="licao",
        duration_target_seconds=60.0,
        beats=[
            ScriptBeat(id=1, title="abertura", text="Eu gravei esse video pra mostrar a esteira."),
            ScriptBeat(id=2, title="custo", text="O custo todo cabe em duzentos reais por mes."),
        ],
    )


def a_board(**kw) -> PlannedStoryboard:
    base = dict(
        beat_id=2, script_anchor="cabe em duzentos reais",
        concept="a wooden desk with a handwritten ledger",
        concept_tags=["desk", "ledger"], sub_shots=2,
        rationale="o argumento central precisa de uma imagem de contabilidade",
    )
    return PlannedStoryboard(beats=[PlannedBeat(**{**base, **kw})])


def test_storyboard_valido_na_primeira(script, anthropic_config, rules, render):
    client = FakeClient([FakeResponse(a_board())])
    board = plan(script, spoken_seconds=10.0, input_hash="h",
                 config=anthropic_config, rules=rules, render=render, client=client)
    assert board.input_hash == "h"
    assert board.n_images == 1
    assert board.beats[0].seconds_per_shot == 2.5
    assert len(client.messages.calls) == 1


def test_ancora_inventado_volta_com_o_texto(script, anthropic_config, rules, render):
    client = FakeClient([
        FakeResponse(a_board(script_anchor="frase que nao esta no roteiro")),
        FakeResponse(a_board()),
    ])
    plan(script, spoken_seconds=10.0, input_hash="h",
         config=anthropic_config, rules=rules, render=render, client=client)

    feedback = client.messages.calls[1]["messages"][-1]["content"]
    assert "nao aparece no texto" in feedback
    assert "frase que nao esta no roteiro" in feedback


def test_sub_shots_demais_volta_com_a_geometria(script, anthropic_config, rules, render):
    client = FakeClient([FakeResponse(a_board(sub_shots=9)), FakeResponse(a_board())])
    plan(script, spoken_seconds=10.0, input_hash="h",
         config=anthropic_config, rules=rules, render=render, client=client)
    assert "geometrico" in client.messages.calls[1]["messages"][-1]["content"]


def test_storyboard_desiste_listando_os_problemas(script, anthropic_config, rules, render):
    ruim = a_board(script_anchor="nada a ver")
    client = FakeClient([FakeResponse(ruim) for _ in range(3)])
    with pytest.raises(StoryboardFailed, match="nao aparece no texto"):
        plan(script, spoken_seconds=10.0, input_hash="h",
             config=anthropic_config, rules=rules, render=render, client=client)
    assert len(client.messages.calls) == 3


def test_o_roteiro_inteiro_vai_no_pedido(script, anthropic_config, rules, render):
    client = FakeClient([FakeResponse(a_board())])
    plan(script, spoken_seconds=10.0, input_hash="h",
         config=anthropic_config, rules=rules, render=render, client=client)
    pedido = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "[beat 1 — abertura]" in pedido
    assert "duzentos reais" in pedido


def test_roteiro_e_cacheado_entre_tentativas(script, anthropic_config, rules, render):
    """A parte grande e estavel do prompt nao muda entre tentativas."""
    client = FakeClient([FakeResponse(a_board(sub_shots=9)), FakeResponse(a_board())])
    plan(script, spoken_seconds=10.0, input_hash="h",
         config=anthropic_config, rules=rules, render=render, client=client)

    primeiro = client.messages.calls[1]["messages"][0]["content"][0]
    assert primeiro["cache_control"] == {"type": "ephemeral"}


def test_prompt_proibe_tempo_absoluto(rules, render):
    p = storyboard_prompt(rules, render, 900.0)
    assert "Voce NAO escreve tempo" in p
    assert "Copie o trecho, nao parafraseie" in p


def test_prompt_diz_quanto_uma_imagem_de_4_sub_planos_ocupa(rules, render):
    """4 sub-planos de 2.5s = 10s de video por imagem gerada."""
    p = storyboard_prompt(rules, render, 900.0)
    assert "10 segundos" in p


def test_prompt_pede_quadro_com_mais_de_um_ponto_de_interesse(rules, render):
    """Porque a imagem vai ser recortada em sub-planos: um objeto solitario no
    centro nao rende quatro planos distintos."""
    p = storyboard_prompt(rules, render, 900.0)
    assert "mais de um ponto de interesse" in p


def test_script_block_mostra_os_ids(script):
    bloco = script_block(script)
    assert "[beat 1 — abertura]" in bloco
    assert "[beat 2 — custo]" in bloco
