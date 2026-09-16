"""Estagio 6: onde cada imagem aprovada cai na fala gravada.

A maioria destes testes verifica o que NAO pode acontecer: imagem no lugar
errado, faixa sobreposta, buraco na timeline, `assets.json` aprovado sendo
reescrito, e o torch sendo carregado quando nao precisa.
"""

from __future__ import annotations

import pytest

from test_bank_reuse import FakeEmbedder

from pipeline import approval
from pipeline.align import aligned_assets, broll_positions, edl_from_spans, spans_for
from pipeline.approval import NotApproved
from pipeline.schemas import (
    AssetEstimate, AssetItem, Assets, BeatPlacement, Placements, Script,
    ScriptBeat, Storyboard, StoryboardBeat, Transcript, TranscriptSegment, Word,
)
from pipeline.stages import align
from pipeline.util import read_json, write_json

SEG = 5.0
N = 24                          # 24 segmentos de 5s = 120s


class Explode:
    """Embedder que falha se tocado. Ver `test_align.py`: o caminho literal
    nao pode carregar o torch."""

    model_name = "explode"

    def embed(self, text):
        raise AssertionError("o embedder foi carregado")


def sem_torch(monkeypatch, embedder=None):
    """Troca o embedder real do estagio por um falso.

    O real importa `sentence_transformers` no primeiro `embed()`, que nao
    esta instalado na suite (e arrastaria o torch se estivesse). Quem passa
    `Explode` esta afirmando que o caminho literal resolve tudo; quem passa
    `FakeEmbedder` quer exercitar o caminho semantico.
    """
    monkeypatch.setattr(
        "pipeline.stages.align.SentenceTransformerEmbedder",
        lambda *a, **k: embedder if embedder is not None else FakeEmbedder())


@pytest.fixture
def cfg(config, tmp_path):
    return config.model_copy(update={"root": tmp_path})


def um_transcript() -> Transcript:
    """24 segmentos, cada um com uma frase reconhecivel pelo indice."""
    segments = []
    for i in range(N):
        start, end = i * SEG, (i + 1) * SEG
        tokens = f"este e o segmento numero {i} da fala gravada".split()
        passo = (end - start) / len(tokens)
        segments.append(TranscriptSegment(
            id=i, start=start, end=end, text=" ".join(tokens),
            words=[Word(start=round(start + j * passo, 3),
                        end=round(start + (j + 1) * passo, 3), word=w)
                   for j, w in enumerate(tokens)],
        ))
    return Transcript(input_hash="audio", language="pt", model="test",
                      duration=N * SEG, segments=segments)


def um_roteiro(ancoras: dict[int, int]) -> Script:
    return Script(
        slug="v1", subject="s", argument="a", viewer_takeaway="t",
        duration_target_seconds=120.0,
        beats=[ScriptBeat(id=b, title="", text=f"este e o segmento numero {s} da fala")
               for b, s in ancoras.items()],
    )


def um_board(ancoras: dict[int, int], *, sub_shots=2, input_hash="board-1") -> Storyboard:
    """Um beat por entrada: `beat_id -> segmento onde o ancora esta`."""
    return Storyboard(input_hash=input_hash, beats=[
        StoryboardBeat(
            beat_id=b, script_anchor=f"segmento numero {s} da fala",
            concept=f"a still life with object {b}",
            concept_tags=[f"object{b}", "still"], sub_shots=sub_shots,
            rationale="porque sim", anchor_offset=0, seconds_per_shot=2.5,
        )
        for b, s in ancoras.items()
    ])


def umas_imagens(board: Storyboard, *, input_hash="assets-1") -> Assets:
    items = [
        AssetItem(beat_id=b.beat_id, origin="generated",
                  path=f"assets/images/{b.beat_id}.png", cost_usd=0.025)
        for b in board.beats
    ]
    return Assets(
        input_hash=input_hash, items=items,
        estimate=AssetEstimate(n_broll=len(items), n_from_bank=0, n_from_stock=0,
                               n_to_generate=len(items), worst_case_usd=0.0,
                               estimated_usd=0.0, budget_usd=3.0, within_budget=True),
        total_cost_usd=0.025 * len(items), by_origin={"generated": len(items)},
    )


@pytest.fixture
def cenario(cfg):
    """Tudo em disco, com as imagens aprovadas. Ancoras nos segmentos 6, 12, 18."""
    ancoras = {1: 6, 2: 12, 3: 18}
    script, board = um_roteiro(ancoras), um_board(ancoras)
    transcript = um_transcript()
    assets = umas_imagens(board)

    work = cfg.work_dir / script.slug
    work.mkdir(parents=True)
    write_json(work / "storyboard.json", board)
    write_json(work / "assets.json", assets)
    approval.grant(work, "storyboard", board.input_hash)
    approval.grant(work, "images", assets.digest())
    return script, board, transcript, assets, work


# --------------------------------------------------------------------------
# o portao
# --------------------------------------------------------------------------


def test_nao_alinha_sem_as_imagens_aprovadas(cfg):
    ancoras = {1: 6}
    script, board = um_roteiro(ancoras), um_board(ancoras)
    assets = umas_imagens(board)
    work = cfg.work_dir / script.slug
    work.mkdir(parents=True)

    with pytest.raises(NotApproved, match="pipeline approve images v1"):
        align.run(script, board, um_transcript(), assets, cfg)


def test_imagens_regeradas_depois_de_aprovar_revogam(cenario, cfg):
    script, board, transcript, assets, work = cenario
    outras = umas_imagens(board, input_hash="assets-2")

    with pytest.raises(NotApproved, match="mudou depois de aprovado"):
        align.run(script, board, transcript, outras, cfg)


# --------------------------------------------------------------------------
# a EDL
# --------------------------------------------------------------------------


def test_a_edl_cobre_a_timeline_sem_buraco_nem_sobreposicao(cenario, cfg):
    """A propriedade que o caminho antigo tinha que VALIDAR e este tem por
    construcao: o a-roll nao e escolhido, e o complemento."""
    script, board, transcript, assets, work = cenario
    edl, _ = align.run(script, board, transcript, assets, cfg)

    assert edl.segments[0].first_segment == 0
    assert edl.segments[-1].last_segment == N - 1
    for anterior, seguinte in zip(edl.segments, edl.segments[1:]):
        assert seguinte.first_segment == anterior.last_segment + 1


def test_as_faixas_de_broll_saem_na_ordem_dos_beats(cenario, cfg):
    script, board, transcript, assets, work = cenario
    edl, _ = align.run(script, board, transcript, assets, cfg)

    broll = [s for s in edl.segments if s.kind == "broll"]
    assert len(broll) == 3
    assert [s.concept for s in broll] == [
        "a still life with object 1", "a still life with object 2",
        "a still life with object 3",
    ]


def test_o_conceito_e_as_tags_viajam_para_a_faixa(cenario, cfg):
    """A EDL tem que descrever a imagem que vai em cima dela, nao so onde ela
    entra: e o que o dry-run mostra e o que uma re-resolucao usaria."""
    script, board, transcript, assets, work = cenario
    edl, _ = align.run(script, board, transcript, assets, cfg)

    primeira = next(s for s in edl.segments if s.kind == "broll")
    assert primeira.concept_tags == ["object1", "still"]


def test_aroll_nao_carrega_conceito(cenario, cfg):
    script, board, transcript, assets, work = cenario
    edl, _ = align.run(script, board, transcript, assets, cfg)

    for segmento in edl.segments:
        if segmento.kind == "aroll":
            assert segmento.concept == "" and segmento.concept_tags == []


def test_sem_beat_alinhado_a_edl_e_um_aroll_so(cfg, monkeypatch):
    """Nenhuma imagem entra, e o video sai inteiro como a-roll — em vez de o
    estagio explodir e voce ficar sem video."""
    ancoras = {1: 6}
    script = um_roteiro(ancoras)
    board = um_board({1: 6})
    board.beats[0].script_anchor = "isto nao aparece em lugar nenhum da fala"
    assets = umas_imagens(board)
    work = cfg.work_dir / script.slug
    work.mkdir(parents=True)
    write_json(work / "assets.json", assets)
    approval.grant(work, "images", assets.digest())
    sem_torch(monkeypatch)

    rigido = cfg.model_copy(deep=True)
    rigido.align.min_similarity = 2.0            # nada passa no semantico
    edl, aligned = align.run(script, board, um_transcript(), assets, rigido)

    assert [s.kind for s in edl.segments] == ["aroll"]
    assert edl.stats.broll_ratio == 0.0


# --------------------------------------------------------------------------
# a traducao beat -> segmento
# --------------------------------------------------------------------------


def test_o_segment_index_aponta_para_a_posicao_na_edl(cenario, cfg):
    """E por POSICAO que o render junta imagem e faixa, nao por
    `first_segment`. Errar aqui poe a imagem do beat 2 na faixa do beat 3 —
    e cada imagem, isolada, continuaria parecendo certa."""
    script, board, transcript, assets, work = cenario
    edl, aligned = align.run(script, board, transcript, assets, cfg)

    posicoes = broll_positions(edl.segments)
    por_beat = {i.beat_id: i.segment_index for i in aligned.items}

    assert sorted(por_beat.values()) == posicoes
    for beat_id, indice in por_beat.items():
        faixa = edl.segments[indice]
        assert faixa.kind == "broll"
        assert faixa.concept == f"a still life with object {beat_id}"


def test_o_assets_aprovado_nao_e_reescrito(cenario, cfg):
    """Reescrever `assets.json` mudaria o que a aprovacao cobre — o proprio
    portao que este estagio acabou de exigir passaria a apontar para outra
    versao. Por isso sai um arquivo novo."""
    script, board, transcript, assets, work = cenario
    antes = (work / "assets.json").read_text(encoding="utf-8")

    align.run(script, board, transcript, assets, cfg)

    assert (work / "assets.json").read_text(encoding="utf-8") == antes
    assert (work / "assets.aligned.json").exists()
    from pipeline import progress
    assert progress.images_gate(work).state == progress.APPROVED


def test_beat_orfao_fica_marcado_em_vez_de_silencioso(cfg, monkeypatch):
    """A imagem existe, foi paga, e nao vai aparecer. Silenciar seria esconder
    dinheiro gasto."""
    ancoras = {1: 6, 2: 12}
    script = um_roteiro(ancoras)
    board = um_board(ancoras)
    board.beats[1].script_anchor = "isto nao existe na fala gravada"
    assets = umas_imagens(board)
    work = cfg.work_dir / script.slug
    work.mkdir(parents=True)
    write_json(work / "assets.json", assets)
    approval.grant(work, "images", assets.digest())
    sem_torch(monkeypatch)

    rigido = cfg.model_copy(deep=True)
    rigido.align.min_similarity = 2.0
    edl, aligned = align.run(script, board, um_transcript(), assets, rigido)

    orfa = next(i for i in aligned.items if i.beat_id == 2)
    assert orfa.segment_index == -1
    assert "nao entra no video" in orfa.note
    assert len([s for s in edl.segments if s.kind == "broll"]) == 1

    relatorio = read_json(work / "align.json", Placements)
    assert relatorio.n_orphans == 1


# --------------------------------------------------------------------------
# o piso do intro
# --------------------------------------------------------------------------


def test_ancora_dentro_do_intro_e_empurrado_e_nao_descartado(cfg):
    """A regra do intro continua valendo, e a imagem aprovada continua
    entrando. Descartar apagaria uma imagem paga por um motivo que nao e
    dela."""
    board = um_board({1: 0})                      # ancora no primeiro segmento
    transcript = um_transcript()
    colocados = [BeatPlacement(beat_id=1, anchor_offset=0, segment=0,
                               method="literal", similarity=1.0)]

    faixas = spans_for(colocados, board, transcript, intro_seconds=20.0)

    assert len(faixas) == 1
    _, inicio, _ = faixas[0]
    assert transcript.segments[inicio].start >= 20.0


def test_sem_piso_a_faixa_comeca_no_ancora(cfg):
    board = um_board({1: 0})
    colocados = [BeatPlacement(beat_id=1, anchor_offset=0, segment=0,
                               method="literal", similarity=1.0)]

    faixas = spans_for(colocados, board, um_transcript(), intro_seconds=0.0)
    assert faixas[0][1] == 0


def test_o_estagio_usa_o_intro_do_config(cenario, cfg):
    script, board, transcript, assets, work = cenario
    edl, _ = align.run(script, board, transcript, assets, cfg)

    primeiro_broll = next(s for s in edl.segments if s.kind == "broll")
    assert primeiro_broll.start >= cfg.editorial.intro_aroll_seconds


# --------------------------------------------------------------------------
# cache e o torch
# --------------------------------------------------------------------------


def test_o_caminho_literal_nao_carrega_o_torch(cenario, cfg, monkeypatch):
    """Todo ancora casa literalmente, entao o embedder nao pode ser tocado —
    ele arrasta centenas de MB de torch."""
    script, board, transcript, assets, work = cenario
    sem_torch(monkeypatch, Explode())

    edl, _ = align.run(script, board, transcript, assets, cfg)
    assert edl.stats.n_broll == 3


def test_segunda_chamada_le_do_disco(cenario, cfg, monkeypatch):
    script, board, transcript, assets, work = cenario
    align.run(script, board, transcript, assets, cfg)

    def explode(*a, **k):
        raise AssertionError("realinhou em vez de ler o artefato")

    monkeypatch.setattr("pipeline.stages.align.place", explode)
    edl, aligned = align.run(script, board, transcript, assets, cfg)
    assert edl.stats.n_broll == 3


def test_a_chave_inclui_o_transcript(cenario, cfg):
    """Regravar muda onde tudo cai, mesmo com o roteiro e as imagens iguais."""
    script, board, transcript, assets, work = cenario
    outro = um_transcript()
    outro.segments[3].text = "esta frase foi dita de outro jeito na segunda tomada"

    assert align.cache_key(board, transcript, assets, cfg) != align.cache_key(
        board, outro, assets, cfg)


# --------------------------------------------------------------------------
# os avisos
# --------------------------------------------------------------------------


def test_cobertura_fora_da_faixa_vira_aviso_e_nao_erro(cenario, cfg):
    """O storyboard estimou a cobertura por `words_per_minute`; a gravacao
    real e que diz quanto deu. Nao e erro de ninguem, e nao pode derrubar o
    estagio depois de as imagens serem pagas."""
    script, board, transcript, assets, work = cenario
    edl, _ = align.run(script, board, transcript, assets, cfg)

    relatorio = read_json(work / "align.json", Placements)
    assert edl.stats.broll_ratio < cfg.editorial.broll_ratio_min
    assert any("abaixo da faixa" in a for a in relatorio.warnings)


def test_nao_reclama_da_duracao_do_broll(cenario, cfg):
    """`broll_min_seconds` nao governa este caminho: a duracao vem de
    `sub_shots * sub_shot_seconds`, que foi aprovado beat por beat. Reclamar
    seria re-litigar decisao tomada, e ensinaria a ignorar aviso."""
    script, board, transcript, assets, work = cenario
    align.run(script, board, transcript, assets, cfg)

    relatorio = read_json(work / "align.json", Placements)
    for aviso in relatorio.warnings:
        assert "minimo" not in aviso and "maximo" not in aviso


def test_o_relatorio_mostra_o_metodo_de_cada_beat(cenario, cfg):
    script, board, transcript, assets, work = cenario
    align.run(script, board, transcript, assets, cfg)
    relatorio = read_json(work / "align.json", Placements)

    texto = align.render_text(relatorio, board)
    assert "beat 1" in texto and "literal" in texto
    assert "segmento numero 6 da fala" in texto


# --------------------------------------------------------------------------
# as funcoes puras
# --------------------------------------------------------------------------


def test_aligned_assets_empareilha_por_ordem():
    """A propriedade e do construtor da EDL: uma faixa de b-roll por span, na
    ordem das spans. Este teste existe para ela nao ser assumida."""
    board = um_board({7: 2, 9: 10})
    transcript = um_transcript()
    faixas = [(7, 2, 3), (9, 10, 11)]
    edl = edl_from_spans(faixas, board, transcript, input_hash="k")

    aligned = aligned_assets(umas_imagens(board), faixas, edl.segments, input_hash="k")
    por_beat = {i.beat_id: i.segment_index for i in aligned.items}

    assert edl.segments[por_beat[7]].first_segment == 2
    assert edl.segments[por_beat[9]].first_segment == 10


def test_o_relatorio_mostra_a_faixa_e_nao_so_o_ancora(cfg):
    """As duas coisas divergem, e a diferenca e o que confunde na revisao: o
    piso do intro empurra a imagem sem mexer no ancora. Achado lendo a saida
    da CLI de verdade, nao o diff.

    Ancora no PRIMEIRO segmento de proposito: e o unico arranjo em que o piso
    do intro tem o que empurrar, e sem ele o teste passaria sem discriminar.
    """
    ancoras = {1: 0}
    script, board = um_roteiro(ancoras), um_board(ancoras)
    assets = umas_imagens(board)
    work = cfg.work_dir / script.slug
    work.mkdir(parents=True)
    write_json(work / "assets.json", assets)
    approval.grant(work, "images", assets.digest())

    align.run(script, board, um_transcript(), assets, cfg)
    relatorio = read_json(work / "align.json", Placements)

    faixa = relatorio.spans[0]
    colocado = relatorio.placements[0]
    assert colocado.segment == 0
    assert faixa.first_segment > 0, "o piso do intro nao empurrou nada"
    assert faixa.start >= cfg.editorial.intro_aroll_seconds

    texto = align.render_text(relatorio, board)
    assert f"{faixa.start:7.1f}s" in texto
    assert "achado no segmento 0" in texto


def test_as_faixas_do_relatorio_casam_com_a_edl(cenario, cfg):
    script, board, transcript, assets, work = cenario
    edl, _ = align.run(script, board, transcript, assets, cfg)
    relatorio = read_json(work / "align.json", Placements)

    broll = [s for s in edl.segments if s.kind == "broll"]
    assert [(f.first_segment, f.last_segment) for f in relatorio.spans] == \
           [(s.first_segment, s.last_segment) for s in broll]
    assert [(f.start, f.end) for f in relatorio.spans] == \
           [(s.start, s.end) for s in broll]
