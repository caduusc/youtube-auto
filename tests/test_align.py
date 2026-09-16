"""Alinhamento beat<->transcript.

Duas propriedades dominam, e as duas sao sobre o que NAO pode acontecer.

O alinhamento e monotonico: o beat 7 nao pode casar antes do beat 6. Um argmax
por ancora, sem cursor, embaralharia as imagens — e o embaralhamento seria
plausivel demais para aparecer numa revisao rapida, porque cada imagem isolada
continuaria "parecendo certa".

E o sinal mais forte vem antes do mais esperto: se voce seguiu o roteiro, as
palavras do ancora estao literalmente na fala, e a busca literal acerta de
graca — sem carregar o torch.
"""

from __future__ import annotations

import math

import pytest

from pipeline.align import find_literal, place, segment_words, spans_for
from pipeline.config import AlignConfig, RenderConfig
from pipeline.schemas import Script, ScriptBeat, Storyboard, StoryboardBeat, Transcript, TranscriptSegment


def transcript(*textos: str, dur: float = 8.0) -> Transcript:
    return Transcript(
        input_hash="h", language="pt", model="small", duration=dur * len(textos),
        segments=[
            TranscriptSegment(id=i, start=i * dur, end=(i + 1) * dur, text=t, words=[])
            for i, t in enumerate(textos)
        ],
    )


def board(*ancoras: str, sub_shots: int = 2) -> Storyboard:
    return Storyboard(input_hash="h", beats=[
        StoryboardBeat(
            beat_id=i + 1, script_anchor=a, concept=f"scene {i}",
            concept_tags=["a", "b"], sub_shots=sub_shots,
            rationale="porque", anchor_offset=i, seconds_per_shot=2.5,
        )
        for i, a in enumerate(ancoras)
    ])


class Explode:
    """Embedder que falha se tocado. Prova que o caminho literal nao o usa."""

    model_name = "nao-deveria-carregar"

    def embed(self, text: str) -> list[float]:
        raise AssertionError("o embedder foi carregado no caminho literal")


class ByWord:
    """Embedder de brinquedo: vetor de contagem de palavras em comum.

    Nao e um modelo — e uma metrica de sobreposicao lexical que se comporta
    como cosseno o bastante para exercitar o caminho semantico.
    """

    model_name = "por-palavra"

    def __init__(self) -> None:
        self.chamadas = 0

    def embed(self, text: str) -> list[float]:
        self.chamadas += 1
        vocab = "custo imagem esteira video reais mes camera luz mesa cabo"
        palavras = set(text.lower().replace(",", "").replace(".", "").split())
        return [1.0 if p in palavras else 0.0 for p in vocab.split()]


# --------------------------------------------------------------------------
# busca literal
# --------------------------------------------------------------------------


def test_literal_acha_dentro_de_um_segmento():
    t = transcript("Eu montei uma esteira que edita video", "sozinha e barata")
    assert find_literal(["edita", "video"], segment_words(t), 0) == 0


def test_literal_atravessa_fronteira_de_segmento():
    """O Whisper quebra por pausa, nao por frase: ancora de seis palavras cai
    partido em dois segmentos com frequencia. Sem atravessar, o caminho literal
    falharia justamente nos ancoras mais longos, que sao os mais confiaveis."""
    t = transcript("Eu montei uma esteira que edita video", "sozinha e ela custa nada")
    palavras = segment_words(t)
    assert find_literal(["edita", "video", "sozinha"], palavras, 0) == 0
    assert find_literal(["video", "sozinha", "e", "ela"], palavras, 0) == 0


def test_literal_respeita_o_cursor():
    t = transcript("o custo cabe em duzentos reais", "e o custo cabe mesmo")
    palavras = segment_words(t)
    assert find_literal(["o", "custo", "cabe"], palavras, 0) == 0
    assert find_literal(["o", "custo", "cabe"], palavras, 1) == 1


def test_literal_nao_acha_o_que_nao_existe():
    t = transcript("uma frase qualquer")
    assert find_literal(["nada", "disso"], segment_words(t), 0) == -1


def test_literal_ignora_acento_e_pontuacao():
    t = transcript("O custo todo cabe em duzentos reais, por mês.")
    assert find_literal(["duzentos", "reais", "por", "mes"], segment_words(t), 0) == 0


def test_ancora_vazio_nao_casa():
    t = transcript("qualquer coisa")
    assert find_literal([], segment_words(t), 0) == -1


# --------------------------------------------------------------------------
# o caminho literal nao carrega o embedder
# --------------------------------------------------------------------------


def test_tudo_literal_nao_toca_no_embedder():
    """Na pratica: se voce seguiu o roteiro, o alinhamento nao paga nada."""
    t = transcript("Eu montei uma esteira que edita video",
                   "e o custo cabe em duzentos reais")
    colocados = place(board("edita video", "duzentos reais"), t,
                      rules=AlignConfig(), embedder=Explode())
    assert [c.method for c in colocados] == ["literal", "literal"]
    assert [c.segment for c in colocados] == [0, 1]
    assert [c.similarity for c in colocados] == [1.0, 1.0]


# --------------------------------------------------------------------------
# monotonicidade
# --------------------------------------------------------------------------


def test_alinhamento_e_monotonico():
    """A frase repetida aparece nos segmentos 0 e 2. O segundo beat tem que
    casar no 2, nao voltar para o 0."""
    t = transcript("o custo cabe", "outra coisa no meio", "o custo cabe")
    colocados = place(board("o custo cabe", "o custo cabe"), t,
                      rules=AlignConfig(), embedder=Explode())
    assert [c.segment for c in colocados] == [0, 2]


def test_segundo_beat_nao_volta_atras_nem_no_caminho_semantico():
    t = transcript("fala sobre camera e luz", "fala sobre mesa e cabo",
                   "fala sobre camera e luz de novo")
    emb = ByWord()
    colocados = place(board("camera luz", "camera luz"), t,
                      rules=AlignConfig(min_similarity=0.1), embedder=emb)
    assert colocados[0].segment < colocados[1].segment, [c.model_dump() for c in colocados]


# --------------------------------------------------------------------------
# orfaos
# --------------------------------------------------------------------------


def test_beat_que_voce_pulou_fica_orfao():
    """Forcar um lugar plausivel para uma imagem que nao tem lugar e pior que
    deixar o segmento em a-roll."""
    t = transcript("falei so disso aqui")
    colocados = place(board("um trecho que eu nunca disse"), t,
                      rules=AlignConfig(min_similarity=0.9), embedder=ByWord())
    assert colocados[0].method == "unaligned"
    assert colocados[0].segment == -1


def test_sem_embedder_o_que_nao_casa_literalmente_fica_orfao():
    t = transcript("falei so disso")
    colocados = place(board("outra coisa"), t, rules=AlignConfig(), embedder=None)
    assert colocados[0].method == "unaligned"


def test_orfao_nao_consome_o_cursor():
    """Um beat orfao no meio nao pode empurrar os seguintes para frente."""
    t = transcript("primeiro trecho falado", "segundo trecho falado")
    colocados = place(
        board("nao existe isso", "segundo trecho"), t,
        rules=AlignConfig(), embedder=None,
    )
    assert colocados[0].method == "unaligned"
    assert colocados[1].segment == 1


# --------------------------------------------------------------------------
# as faixas: fronteira sempre em fronteira de segmento
# --------------------------------------------------------------------------


def test_faixa_cresce_em_segmento_inteiro():
    """4 sub-planos de 2.5s = 10s; segmentos de 8s -> precisa de dois."""
    t = transcript("a", "b", "c", "d", dur=8.0)
    b = board("a", sub_shots=4)
    colocados = place(b, t, rules=AlignConfig(), embedder=Explode())
    assert spans_for(colocados, b, t) == [(1, 0, 1)]


def test_faixa_de_um_segmento_quando_ja_cobre():
    """2 sub-planos de 2.5s = 5s; um segmento de 8s basta."""
    t = transcript("a", "b", dur=8.0)
    b = board("a", sub_shots=2)
    colocados = place(b, t, rules=AlignConfig(), embedder=Explode())
    assert spans_for(colocados, b, t) == [(1, 0, 0)]


def test_faixas_nao_se_sobrepoem():
    """Duas imagens no mesmo segmento: a segunda e empurrada para depois."""
    t = transcript("o custo cabe aqui", "e tambem o custo cabe", "fim", dur=8.0)
    b = board("o custo cabe", "o custo cabe", sub_shots=2)
    colocados = place(b, t, rules=AlignConfig(), embedder=Explode())
    faixas = spans_for(colocados, b, t)
    assert faixas == [(1, 0, 0), (2, 1, 1)]


def test_orfao_nao_gera_faixa():
    t = transcript("so isso")
    b = board("nao existe")
    colocados = place(b, t, rules=AlignConfig(), embedder=None)
    assert spans_for(colocados, b, t) == []


def test_faixa_no_fim_do_video_nao_estoura():
    """Beat que pede 10s no ultimo segmento de 8s: para no ultimo."""
    t = transcript("a", "b", dur=8.0)
    b = board("b", sub_shots=4)
    colocados = place(b, t, rules=AlignConfig(), embedder=Explode())
    assert spans_for(colocados, b, t) == [(1, 1, 1)]
