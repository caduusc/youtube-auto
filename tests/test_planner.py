"""Loop de rejeicao do estagio 3. A API e mockada: nenhum teste gasta nada."""

from __future__ import annotations

import pytest
from conftest import alternating, span

from pipeline.planner import PlanningFailed, plan, retry_request, system_prompt
from pipeline.schemas import PlannedEDL


class FakeUsage:
    input_tokens = 5000
    output_tokens = 3000
    cache_read_input_tokens = 0


class FakeResponse:
    def __init__(self, parsed, stop_reason="end_turn", stop_details=None):
        self.parsed_output = parsed
        self.stop_reason = stop_reason
        self.stop_details = stop_details
        self.usage = FakeUsage()
        self.content = [{"type": "text", "text": "{}"}]


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


def bad_edl():
    """b-roll dentro dos primeiros 20s e curto demais."""
    return PlannedEDL(spans=[span("broll", 0, 0), span("aroll", 1, 199)])


def good_edl():
    return alternating(aroll_len=4, broll_len=5)


# --------------------------------------------------------------------------


def test_aceita_edl_valida_na_primeira(transcript, config):
    client = FakeClient([FakeResponse(good_edl())])
    edl = plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    assert edl.attempts == 1
    assert len(client.messages.calls) == 1
    assert edl.input_hash == transcript.digest()


def test_rejeita_e_tenta_de_novo_com_o_erro_especifico(transcript, config):
    client = FakeClient([FakeResponse(bad_edl()), FakeResponse(good_edl())])
    edl = plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    assert edl.attempts == 2
    assert len(client.messages.calls) == 2

    # a segunda chamada carrega o feedback com a violacao nomeada
    followup = client.messages.calls[1]["messages"][-1]["content"]
    assert "primeiros 20s" in followup
    assert "minimo de 8s" in followup


def test_desiste_depois_de_max_attempts(transcript, config):
    client = FakeClient([FakeResponse(bad_edl()) for _ in range(3)])
    with pytest.raises(PlanningFailed, match="nao produziu EDL valida em 3 tentativas"):
        plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    assert len(client.messages.calls) == 3


def test_indice_invalido_volta_como_feedback(transcript, config):
    """Faixa apontando para fora do transcript nao pode derrubar o estagio:
    e exatamente o tipo de erro que o modelo sabe corrigir."""
    broken = PlannedEDL(spans=[span("aroll", 0, 5), span("broll", 6, 9999)])
    client = FakeClient([FakeResponse(broken), FakeResponse(good_edl())])
    edl = plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    assert edl.attempts == 2
    assert "transcript tem indices" in client.messages.calls[1]["messages"][-1]["content"]


def test_recusa_da_api_nao_vira_retry(transcript, config):
    class Details:
        category = "cyber"

    client = FakeClient([FakeResponse(None, stop_reason="refusal", stop_details=Details())])
    with pytest.raises(PlanningFailed, match="recusou a requisicao"):
        plan(transcript, config=config.anthropic, rules=config.editorial, client=client)


def test_resposta_truncada_aponta_para_o_config(transcript, config):
    client = FakeClient([FakeResponse(None, stop_reason="max_tokens")])
    with pytest.raises(PlanningFailed, match="anthropic.max_tokens"):
        plan(transcript, config=config.anthropic, rules=config.editorial, client=client)


# --------------------------------------------------------------------------
# o prompt
# --------------------------------------------------------------------------


def test_regras_do_prompt_saem_do_config(config):
    """As regras editoriais tem uma fonte so: se o config e o prompt
    divergirem, o modelo tenta acertar um alvo que o validador nao aceita."""
    prompt = system_prompt(config.editorial)
    rules = config.editorial
    assert f"{rules.intro_aroll_seconds:.0f} segundos sao `aroll`" in prompt
    assert f"minimo {rules.broll_min_seconds:.0f}" in prompt
    assert f"maximo {rules.broll_max_seconds:.0f}" in prompt
    assert f"{rules.max_switches_per_minute} trocas de tela por minuto" in prompt
    assert f"{rules.max_switches_per_minute + 1} trocas dentro de uma janela" in prompt
    assert f"entre {rules.broll_ratio_min:.0%} e {rules.broll_ratio_max:.0%}" in prompt


def test_prompt_proibe_texto_e_rosto_na_imagem(config):
    prompt = system_prompt(config.editorial)
    assert "Sem texto na imagem" in prompt
    assert "Sem rosto em primeiro plano" in prompt


def test_transcript_e_cacheado_entre_tentativas(transcript, config):
    client = FakeClient([FakeResponse(bad_edl()), FakeResponse(good_edl())])
    plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    first = client.messages.calls[0]
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # a mesma mensagem inicial segue na segunda chamada, com o prefixo intacto
    assert client.messages.calls[1]["messages"][0] == first["messages"][0]


def test_usa_o_modelo_e_o_effort_do_config(transcript, config):
    client = FakeClient([FakeResponse(good_edl())])
    plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    call = client.messages.calls[0]
    assert call["model"] == config.anthropic.model == "claude-opus-5"
    assert call["output_config"] == {"effort": config.anthropic.effort}
    assert call["output_format"] is PlannedEDL


def test_retry_request_lista_todas_as_violacoes():
    text = retry_request(["erro um", "erro dois"])
    assert "- erro um" in text and "- erro dois" in text
