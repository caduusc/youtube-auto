"""Loop de rejeicao do estagio 3. A API e mockada: nenhum teste gasta nada."""

from __future__ import annotations

import pytest
from conftest import alternating, span

from pipeline.planner import (
    BRIEF_PROMPT, PlanningFailed, brief_block, plan, retry_request, system_prompt,
)
from pipeline.schemas import Brief, PlannedEDL


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


def a_brief() -> Brief:
    return Brief(
        subject="por que metodo de produtividade falha sem clareza de objetivo",
        argument="sem saber onde quer chegar, qualquer sistema vira ritual",
        audience_takeaway="defina o destino antes de escolher a ferramenta",
        visual_vocabulary=[
            "a worn leather notebook on a wooden table", "morning window light",
            "a single wooden chair", "hand-drawn map on paper",
            "a compass beside a mug", "empty corkboard with pins",
        ],
        avoid=["chessboard", "stock photo handshake", "futuristic hologram"],
    )


def brief_first(*edl_responses):
    """A fase 1 consome uma resposta antes das tentativas de EDL."""
    return FakeClient([FakeResponse(a_brief()), *edl_responses])


def bad_edl():
    """b-roll dentro dos primeiros 20s e curto demais."""
    return PlannedEDL(spans=[span("broll", 0, 0), span("aroll", 1, 199)])


def good_edl():
    return alternating(aroll_len=4, broll_len=5)


# --------------------------------------------------------------------------


def test_aceita_edl_valida_na_primeira(transcript, config):
    client = brief_first(FakeResponse(good_edl()))
    edl = plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    assert edl.attempts == 1
    assert len(client.messages.calls) == 2      # briefing + EDL
    assert edl.input_hash == transcript.digest()


def test_rejeita_e_tenta_de_novo_com_o_erro_especifico(transcript, config):
    client = brief_first(FakeResponse(bad_edl()), FakeResponse(good_edl()))
    edl = plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    assert edl.attempts == 2
    assert len(client.messages.calls) == 3      # briefing + 2 tentativas

    # a chamada seguinte carrega o feedback com a violacao nomeada
    followup = client.messages.calls[2]["messages"][-1]["content"]
    assert "primeiros 20s" in followup
    assert "minimo de 8s" in followup


def test_desiste_depois_de_max_attempts(transcript, config):
    client = brief_first(*[FakeResponse(bad_edl()) for _ in range(3)])
    with pytest.raises(PlanningFailed, match="nao produziu EDL valida em 3 tentativas"):
        plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    assert len(client.messages.calls) == 4      # briefing + 3 tentativas


def test_indice_invalido_volta_como_feedback(transcript, config):
    """Faixa apontando para fora do transcript nao pode derrubar o estagio:
    e exatamente o tipo de erro que o modelo sabe corrigir."""
    broken = PlannedEDL(spans=[span("aroll", 0, 5), span("broll", 6, 9999)])
    client = brief_first(FakeResponse(broken), FakeResponse(good_edl()))
    edl = plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    assert edl.attempts == 2
    assert "transcript tem indices" in client.messages.calls[2]["messages"][-1]["content"]


def test_recusa_da_api_nao_vira_retry(transcript, config):
    class Details:
        category = "cyber"

    client = brief_first(FakeResponse(None, stop_reason="refusal", stop_details=Details()))
    with pytest.raises(PlanningFailed, match="recusou a requisicao"):
        plan(transcript, config=config.anthropic, rules=config.editorial, client=client)


def test_resposta_truncada_aponta_para_o_config(transcript, config):
    client = brief_first(FakeResponse(None, stop_reason="max_tokens"))
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
    client = brief_first(FakeResponse(bad_edl()), FakeResponse(good_edl()))
    plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    first = client.messages.calls[1]           # a primeira tentativa de EDL
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # a mesma mensagem inicial segue na tentativa seguinte, com o prefixo intacto
    assert client.messages.calls[2]["messages"][0] == first["messages"][0]


def test_usa_o_modelo_e_o_effort_do_config(transcript, config):
    client = brief_first(FakeResponse(good_edl()))
    plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    for call in client.messages.calls:
        assert call["model"] == config.anthropic.model == "claude-opus-5"
        assert call["output_config"] == {"effort": config.anthropic.effort}
    assert client.messages.calls[0]["output_format"] is Brief
    assert client.messages.calls[1]["output_format"] is PlannedEDL


def test_retry_request_lista_todas_as_violacoes():
    text = retry_request(["erro um", "erro dois"])
    assert "- erro um" in text and "- erro dois" in text


# --------------------------------------------------------------------------
# fase 1: o briefing visual
# --------------------------------------------------------------------------


def test_briefing_roda_antes_da_edl(transcript, config):
    """As duas tarefas competiam na mesma resposta. Separar faz a primeira
    chamada ler o video e a segunda decidir cortes com o vocabulario pronto."""
    client = brief_first(FakeResponse(good_edl()))
    plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    assert client.messages.calls[0]["output_format"] is Brief
    assert client.messages.calls[1]["output_format"] is PlannedEDL


def test_briefing_recebe_a_transcricao_inteira(transcript, config):
    client = brief_first(FakeResponse(good_edl()))
    plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    texto = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    for segment in (transcript.segments[0], transcript.segments[-1]):
        assert segment.text in texto


def test_briefing_nao_decide_tempos(config):
    """A fase 1 nao pode escolher cortes: e o que fazia a visual sofrer."""
    assert "Nao decida cortes nem tempos aqui" in BRIEF_PROMPT
    assert "UNIDADE" in BRIEF_PROMPT


def test_vocabulario_do_briefing_chega_na_fase_2(transcript, config):
    client = brief_first(FakeResponse(good_edl()))
    plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    pedido = client.messages.calls[1]["messages"][0]["content"][0]["text"]
    for elemento in a_brief().visual_vocabulary:
        assert elemento in pedido
    for clice in a_brief().avoid:
        assert clice in pedido


def test_prompt_da_fase_2_mantem_a_ancora_local(config):
    """O briefing define o vocabulario, nao substitui a relevancia local:
    imagem que ilustra o tema mas nao o trecho e pior que generica."""
    prompt = system_prompt(config.editorial)
    assert "vocabulario" in prompt.lower()
    assert "ancorada ao proprio" in prompt
    assert "descolamento" in prompt


def test_briefing_fica_gravado_na_edl(transcript, config):
    """Para voce ver no dry-run o que guiou as imagens."""
    client = brief_first(FakeResponse(good_edl()))
    edl = plan(transcript, config=config.anthropic, rules=config.editorial, client=client)

    assert edl.brief is not None
    assert edl.brief.subject == a_brief().subject


def test_uma_fase_continua_possivel(transcript, config):
    """`two_phase=False` pula o briefing: uma chamada a menos por video."""
    client = FakeClient([FakeResponse(good_edl())])
    edl = plan(transcript, config=config.anthropic, rules=config.editorial,
               client=client, two_phase=False)

    assert len(client.messages.calls) == 1
    assert edl.brief is None


def test_recusa_no_briefing_nao_vira_retry(transcript, config):
    class Details:
        category = "cyber"

    client = FakeClient([FakeResponse(None, stop_reason="refusal", stop_details=Details())])
    with pytest.raises(PlanningFailed, match="recusou o briefing"):
        plan(transcript, config=config.anthropic, rules=config.editorial, client=client)


def test_brief_block_lista_vocabulario_e_evitar():
    texto = brief_block(a_brief())
    assert "vocabulario visual" in texto
    assert "evitar neste video" in texto
    assert "chessboard" in texto
