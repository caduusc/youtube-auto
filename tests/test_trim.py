"""Deteccao de trecho morto.

O caso que domina este arquivo e negativo: em portugues, "e" e o verbo "ser"
e "um" e artigo. Cortar por token destruiria frases, entao a maior parte dos
testes verifica o que NAO deve ser cortado.
"""

from __future__ import annotations

import pytest

from pipeline.config import TrimConfig
from pipeline.schemas import Transcript, TranscriptSegment, Word
from pipeline.trim import build_plan, find_cuts, normalize, remap_transcript


def fala(*tokens: tuple[str, float, float]) -> Transcript:
    """Transcript de uma frase, a partir de (palavra, inicio, fim)."""
    words = [Word(start=s, end=e, word=w) for w, s, e in tokens]
    return Transcript(
        input_hash="h", language="pt", model="small",
        duration=words[-1].end + 0.5,
        segments=[TranscriptSegment(
            id=0, start=words[0].start, end=words[-1].end,
            text=" ".join(w.word for w in words), words=words,
        )],
    )


FPS = 30.0


@pytest.fixture
def rules():
    return TrimConfig()


# --------------------------------------------------------------------------
# o que NAO pode ser cortado
# --------------------------------------------------------------------------


def test_verbo_e_em_fala_corrida_nao_e_cortado(rules):
    """'isso é muito importante' — cortar o 'é' destroi a frase."""
    t = fala(("isso", 0.00, 0.30), ("é", 0.30, 0.42),
             ("muito", 0.42, 0.75), ("importante", 0.75, 1.40))
    assert find_cuts(t, rules) == []


def test_verbo_e_com_enfase_nao_e_cortado(rules):
    """Fala pausada com enfase no verbo: duracao sobe, mas nao o suficiente
    e o silencio ao redor continua pequeno."""
    t = fala(("o", 0.00, 0.12), ("problema", 0.12, 0.60),
             ("é", 0.70, 0.98), ("esse", 1.03, 1.40))
    assert find_cuts(t, rules) == []


def test_artigo_um_nao_e_cortado(rules):
    t = fala(("tem", 0.00, 0.25), ("um", 0.25, 0.40),
             ("problema", 0.40, 0.95), ("serio", 0.95, 1.40))
    assert find_cuts(t, rules) == []


def test_pausa_curta_de_respiracao_nao_e_cortada(rules):
    t = fala(("primeira", 0.00, 0.50), ("frase", 0.50, 0.90),
             ("segunda", 1.35, 1.90), ("frase", 1.90, 2.30))
    assert find_cuts(t, rules) == []   # gap de 0.45s, abaixo de 0.8


# --------------------------------------------------------------------------
# o que DEVE ser cortado
# --------------------------------------------------------------------------


def test_hesitacao_alongada_com_silencio_e_cortada(rules):
    t = fala(("entao", 0.00, 0.40), ("é", 0.80, 1.45), ("eu", 1.90, 2.10),
             ("acho", 2.10, 2.45))
    cuts = find_cuts(t, rules)
    fillers = [c for c in cuts if c.reason == "filler"]
    assert len(fillers) == 1
    assert fillers[0].token == "é"
    assert fillers[0].start == pytest.approx(0.80)
    assert fillers[0].end == pytest.approx(1.45)


def test_pausa_longa_e_cortada_com_margem(rules):
    t = fala(("acabou", 0.00, 0.50), ("a", 2.00, 2.10), ("frase", 2.10, 2.60))
    pausas = [c for c in find_cuts(t, rules) if c.reason == "pause"]
    assert len(pausas) == 1
    # sobra margem nas duas pontas, para a emenda nao soar rente
    assert pausas[0].start == pytest.approx(0.50 + rules.keep_margin_seconds)
    assert pausas[0].end == pytest.approx(2.00 - rules.keep_margin_seconds)


def test_hm_com_silencio_e_cortado(rules):
    t = fala(("bom", 0.00, 0.30), ("hm", 0.70, 1.25), ("vamos", 1.70, 2.10))
    assert any(c.reason == "filler" and c.token == "hm" for c in find_cuts(t, rules))


def test_variantes_de_acento_casam(rules):
    """O Whisper alterna entre 'é', 'eh' e 'ehh' para o mesmo som."""
    for token in ("é", "eh", "ehh", "Eh"):
        t = fala(("bom", 0.00, 0.30), (token, 0.70, 1.25), ("vamos", 1.70, 2.10))
        assert any(c.reason == "filler" for c in find_cuts(t, rules)), token


@pytest.mark.parametrize("token,esperado", [
    ("é", "e"), ("Eh!", "eh"), ("hmm...", "hmm"), ("não", "nao"), ("um,", "um"),
])
def test_normalize(token, esperado):
    assert normalize(token) == esperado


# --------------------------------------------------------------------------
# fusao e trechos mantidos
# --------------------------------------------------------------------------


def test_hesitacao_cercada_de_pausa_vira_um_corte_so(rules):
    t = fala(("entao", 0.00, 0.40), ("é", 1.50, 2.20), ("eu", 3.40, 3.60))
    cuts = find_cuts(t, rules)
    assert len(cuts) == 1, [c.model_dump() for c in cuts]
    assert cuts[0].reason == "filler"      # a informacao util para revisar
    assert cuts[0].start == pytest.approx(0.52)
    assert cuts[0].end == pytest.approx(3.28)


def test_trechos_mantidos_sao_o_complemento_dos_cortes(rules):
    t = fala(("para", 0.00, 0.50), ("dois", 3.00, 3.50), ("tres", 3.50, 4.00))
    plan = build_plan(t, rules, FPS)
    assert len(plan.keep) == 2
    assert plan.keep[0].start == 0.0
    assert plan.keep[1].end == pytest.approx(t.duration)
    assert plan.stats.removed_seconds > 0


def test_fragmento_curto_entre_cortes_e_absorvido(rules):
    """Um trecho de 0.1s entre dois cortes viraria um estalo."""
    t = fala(("xis", 0.00, 0.40), ("be", 2.00, 2.10), ("ce", 4.00, 4.40))
    plan = build_plan(t, rules, FPS)
    assert all(r.duration >= rules.min_keep_seconds for r in plan.keep)


def test_estatisticas_fecham(rules):
    t = fala(("para", 0.00, 0.50), ("dois", 3.00, 3.50))
    plan = build_plan(t, rules, FPS)
    s = plan.stats
    # as estatisticas sao arredondadas a 3 decimais para leitura, e um frame
    # a 30fps nao e numero redondo (1/30 = 0.0333...)
    assert s.trimmed_seconds == pytest.approx(sum(r.duration for r in plan.keep), abs=1e-3)
    assert s.removed_seconds == pytest.approx(s.original_seconds - s.trimmed_seconds, abs=1e-3)
    assert s.n_cuts == s.n_pause_cuts + s.n_filler_cuts


# --------------------------------------------------------------------------
# desligado
# --------------------------------------------------------------------------


def test_desligado_mantem_tudo():
    rules = TrimConfig(enabled=False)
    t = fala(("para", 0.00, 0.50), ("dois", 5.00, 5.50))
    plan = build_plan(t, rules, FPS)
    assert plan.enabled is False
    assert plan.cuts == []
    assert len(plan.keep) == 1
    assert plan.stats.removed_seconds == 0.0


def test_regras_diferentes_dao_chaves_de_cache_diferentes():
    t = fala(("para", 0.00, 0.50), ("dois", 3.00, 3.50))
    a = build_plan(t, TrimConfig(), FPS)
    b = build_plan(t, TrimConfig(pause_min_seconds=0.5), FPS)
    assert a.input_hash != b.input_hash


# --------------------------------------------------------------------------
# transcript remapeado
# --------------------------------------------------------------------------


def test_remap_desloca_os_tempos(rules):
    t = fala(("para", 0.00, 0.50), ("dois", 3.00, 3.50), ("tres", 3.50, 4.00))
    plan = build_plan(t, rules, FPS)
    novo = remap_transcript(t, plan)

    assert novo.duration == pytest.approx(plan.stats.trimmed_seconds, abs=1e-3)
    tokens = [w.word for s in novo.segments for w in s.words]
    assert tokens == ["para", "dois", "tres"]
    # "dois" comecava em 3.00 e agora vem logo depois de "para"
    dois = [w for s in novo.segments for w in s.words if w.word == "dois"][0]
    assert dois.start < 1.0


def test_palavra_dentro_do_corte_desaparece(rules):
    t = fala(("entao", 0.00, 0.40), ("é", 0.80, 1.45), ("eu", 1.90, 2.10))
    novo = remap_transcript(t, build_plan(t, rules, FPS))
    tokens = [w.word for s in novo.segments for w in s.words]
    assert "é" not in tokens
    assert tokens == ["entao", "eu"]


def test_indices_sao_renumerados_sem_buraco(rules):
    """O estagio 3 referencia segmento por indice: um buraco quebraria a EDL."""
    words_a = [Word(start=0.0, end=0.4, word="ola")]
    words_b = [Word(start=2.0, end=2.6, word="é")]      # segmento inteiro cortado
    words_c = [Word(start=5.0, end=5.5, word="tchau")]
    t = Transcript(
        input_hash="h", language="pt", model="small", duration=6.0,
        segments=[
            TranscriptSegment(id=0, start=0.0, end=0.4, text="ola", words=words_a),
            TranscriptSegment(id=1, start=2.0, end=2.6, text="é", words=words_b),
            TranscriptSegment(id=2, start=5.0, end=5.5, text="tchau", words=words_c),
        ],
    )
    novo = remap_transcript(t, build_plan(t, rules, FPS))
    assert [s.id for s in novo.segments] == list(range(len(novo.segments)))
    assert [s.text for s in novo.segments] == ["ola", "tchau"]


def test_remap_sem_corte_preserva_o_transcript():
    rules = TrimConfig(enabled=False)
    t = fala(("para", 0.00, 0.50), ("dois", 0.50, 1.00))
    novo = remap_transcript(t, build_plan(t, rules, FPS))
    assert [w.start for s in novo.segments for w in s.words] == [0.0, 0.5]


# --------------------------------------------------------------------------
# revisabilidade
# --------------------------------------------------------------------------


def test_corte_carrega_o_texto_ao_redor(rules):
    t = fala(("isso", 0.00, 0.30), ("aqui", 0.30, 0.60), ("é", 1.00, 1.70),
             ("bem", 2.10, 2.40), ("claro", 2.40, 2.80))
    cut = [c for c in find_cuts(t, rules) if c.reason == "filler"][0]
    assert "[é]" in cut.context          # o token cortado, marcado
    assert "aqui" in cut.context         # e o contexto para julgar
    assert "bem" in cut.context


# --------------------------------------------------------------------------
# bordas da gravacao
# --------------------------------------------------------------------------


def test_filler_como_ultima_palavra_e_detectado(rules):
    """O silencio depois da ultima palavra e medido contra o fim do audio.
    Uma variavel sombreada aqui fazia esse caso usar a duracao da palavra
    anterior em vez da duracao do video."""
    words = [Word(start=0.0, end=0.4, word="bom"), Word(start=1.0, end=1.6, word="hm")]
    t = Transcript(
        input_hash="h", language="pt", model="small", duration=3.1,
        segments=[TranscriptSegment(id=0, start=0.0, end=1.6, text="bom hm", words=words)],
    )
    assert any(c.reason == "filler" and c.token == "hm" for c in find_cuts(t, rules))


def test_palavra_da_lista_abrindo_o_video_sem_silencio_nao_e_cortada(rules):
    """'Um dos problemas...' abrindo o video. Tratar a borda como silencio
    infinito cortaria o artigo."""
    t = fala(("um", 0.00, 0.45), ("dos", 0.45, 0.70),
             ("problemas", 0.70, 1.30), ("serios", 1.30, 1.80))
    assert find_cuts(t, rules) == []


def test_pausa_retorica_depois_do_verbo_nao_corta_o_verbo(rules):
    """'o problema é... que ninguém olha' — o silencio esta DEPOIS do verbo,
    e o teste olha o lado de antes, entao a frase mantem o verbo."""
    t = fala(("o", 0.00, 0.15), ("problema", 0.15, 0.75), ("é", 0.75, 1.20),
             ("que", 2.10, 2.30), ("ninguem", 2.30, 2.80), ("olha", 2.80, 3.20))
    cuts = find_cuts(t, rules)
    assert not any(c.reason == "filler" for c in cuts), [c.model_dump() for c in cuts]
    # a pausa em si ainda e cortada, o que e desejavel
    assert any(c.reason == "pause" for c in cuts)


def test_trechos_mantidos_ficam_no_grid_de_frames(rules):
    """Medido: 20 cortes em tempos nao alinhados dao 24ms de dessincronia
    entre audio e video, porque `atrim` corta por amostra e `trim` por frame.
    Alinhado, da zero."""
    t = fala(("para", 0.00, 0.47), ("dois", 3.13, 3.61), ("tres", 3.61, 4.07))
    plan = build_plan(t, rules, FPS)
    for span in plan.keep:
        assert abs(span.start * FPS - round(span.start * FPS)) < 1e-6, span.start
        assert abs(span.end * FPS - round(span.end * FPS)) < 1e-6, span.end


def test_fps_diferente_da_chave_de_cache_diferente(rules):
    """O plano depende do grid de frames, entao trocar o fps de saida
    invalida o corte."""
    t = fala(("para", 0.00, 0.50), ("dois", 3.00, 3.50))
    assert build_plan(t, rules, 30.0).input_hash != build_plan(t, rules, 60.0).input_hash


# --------------------------------------------------------------------------
# a assimetria do silencio, e o piso relativo que ela exige
# --------------------------------------------------------------------------


def test_hesitacao_com_silencio_so_antes_e_cortada(rules):
    """A forma real de uma hesitacao: pausa, som, e a fala retoma rente.

    O intervalo DEPOIS vem exatamente 0.00s do faster-whisper em quase todo
    candidato, porque o modelo atribui spans contiguos. Exigir aquele lado
    fazia esta regra nunca disparar em transcript nenhum.
    """
    t = fala(("entao", 0.00, 0.40), ("é", 1.00, 1.75), ("eu", 1.75, 1.95),
             ("acho", 1.95, 2.30), ("que", 2.30, 2.50), ("sim", 2.50, 2.90))
    fillers = [c for c in find_cuts(t, rules) if c.reason == "filler"]
    assert [c.token for c in fillers] == ["é"]


def test_verbo_abrindo_frase_depois_de_pausa_nao_e_cortado(rules):
    """'...olha isso. É importante que' — tem silencio antes e esta na lista.

    E o caso que a regra de um lado so poderia destruir, e o que segura e a
    duracao: 0.30s e o tamanho normal de um "é" nesta fala, nao de um
    alongamento. Sem o piso relativo, `filler_min_seconds` sozinho nao
    distingue os dois.
    """
    t = fala(("olha", 0.00, 0.35), ("isso", 0.35, 0.75),
             ("é", 1.50, 1.95),                            # 0.45s, > o piso absoluto
             ("importante", 1.95, 2.60), ("que", 2.60, 2.80),
             ("é", 2.80, 3.10), ("assim", 3.10, 3.50),     # mediana do "é" fica em 0.30s
             ("e", 3.50, 3.72), ("pronto", 3.72, 4.20))
    fillers = [c for c in find_cuts(t, rules) if c.reason == "filler"]
    assert fillers == [], [c.model_dump() for c in fillers]


def test_alongamento_acima_da_mediana_da_propria_fala_e_cortado(rules):
    """A mesma fala do teste anterior, com o primeiro 'é' alongado ao dobro.

    Nada mudou no config: o que mudou e a duracao relativa ao que aquele
    token dura no resto da gravacao.
    """
    t = fala(("olha", 0.00, 0.35), ("isso", 0.35, 0.75),
             ("é", 1.50, 2.35),                            # 0.85s, ~2.8x a mediana
             ("importante", 2.35, 3.00), ("que", 3.00, 3.20),
             ("é", 3.20, 3.50), ("assim", 3.50, 3.90),
             ("e", 3.90, 4.12), ("pronto", 4.12, 4.60))
    fillers = [c for c in find_cuts(t, rules) if c.reason == "filler"]
    assert [c.token for c in fillers] == ["é"]
    assert fillers[0].start == pytest.approx(1.50)


def test_token_raro_cai_no_piso_absoluto(rules):
    """'ahn' aparece uma vez e nao tem mediana.

    Com amostra de um, a mediana seria a propria palavra e a razao daria 1.0,
    o que nunca cortaria. Token sem amostra usa o piso absoluto — e para os
    inequivocos ele basta, porque nenhum deles e palavra de conteudo.
    """
    t = fala(("bom", 0.00, 0.30), ("ahn", 0.70, 1.25), ("vamos", 1.25, 1.65))
    assert any(c.token == "ahn" for c in find_cuts(t, rules) if c.reason == "filler")


def test_mediana_ignora_amostra_pequena(rules):
    from pipeline.trim import token_medians

    words = [Word(start=0.0, end=0.2, word="e"), Word(start=0.3, end=0.5, word="e"),
             Word(start=0.6, end=1.0, word="so")]
    assert token_medians(words) == {}          # duas e uma ocorrencia, nenhuma serve

    words.append(Word(start=1.1, end=1.7, word="e"))
    assert token_medians(words)["e"] == pytest.approx(0.2)   # 0.2, 0.2, 0.6


def test_mediana_nao_e_media(rules):
    """Uma hesitacao longa nao pode levantar o piso e se proteger sozinha."""
    from pipeline.trim import token_medians

    words = [Word(start=i * 0.5, end=i * 0.5 + 0.2, word="e") for i in range(5)]
    words.append(Word(start=9.0, end=9.9, word="e"))        # a hesitacao
    assert token_medians(words)["e"] == pytest.approx(0.2)   # media daria 0.317



# --------------------------------------------------------------------------
# aperto de pausa: o mecanismo para fala fluente
# --------------------------------------------------------------------------


def test_aperto_desligado_por_default(rules):
    """Muitas emendas e um custo real, entao o default nao liga sozinho."""
    assert rules.pause_max_seconds == 0.0
    t = fala(("uma", 0.00, 0.30), ("frase", 0.75, 1.20))   # gap de 0.45s
    assert find_cuts(t, rules) == []


def test_aperto_deixa_a_pausa_no_tamanho_do_teto():
    rules = TrimConfig(pause_max_seconds=0.20)
    t = fala(("uma", 0.00, 0.30), ("frase", 0.80, 1.20))   # gap de 0.50s
    cuts = find_cuts(t, rules)
    assert [c.reason for c in cuts] == ["squeeze"]
    # sobra exatamente o teto, nao zero: a pausa continua existindo
    assert cuts[0].duration == pytest.approx(0.50 - 0.20)


def test_aperto_tira_do_meio_do_intervalo():
    """De uma ponta so, uma das duas palavras fica emendada rente."""
    rules = TrimConfig(pause_max_seconds=0.20)
    t = fala(("uma", 0.00, 0.30), ("frase", 0.80, 1.20))
    cut = find_cuts(t, rules)[0]
    assert cut.start - 0.30 == pytest.approx(0.10)   # sobra 0.10s de cada lado
    assert 0.80 - cut.end == pytest.approx(0.10)


def test_pausa_longa_ainda_e_removida_e_nao_apertada():
    """O teto nao rebaixa a remocao: acima de `pause_min_seconds` continua
    saindo quase inteira, que e a batida audivel."""
    rules = TrimConfig(pause_max_seconds=0.20, pause_min_seconds=0.8)
    t = fala(("acabou", 0.00, 0.50), ("segue", 2.00, 2.60))
    cuts = find_cuts(t, rules)
    assert [c.reason for c in cuts] == ["pause"]
    assert cuts[0].duration == pytest.approx(1.50 - 2 * rules.keep_margin_seconds)


def test_intervalo_abaixo_do_teto_nao_e_tocado():
    rules = TrimConfig(pause_max_seconds=0.30)
    t = fala(("uma", 0.00, 0.30), ("frase", 0.55, 1.00))   # gap de 0.25s
    assert find_cuts(t, rules) == []


def test_aperto_rende_mais_que_remocao_em_fala_fluente():
    """O caso que motivou o mecanismo: intervalos medios, nenhum longo.

    Seis intervalos de 0.5s. A remocao com limiar 0.8 nao pega nenhum; o
    aperto em 0.2 tira 0.3s de cada um.
    """
    tokens = [(f"p{i}", i * 1.0, i * 1.0 + 0.5) for i in range(7)]
    t = fala(*tokens)                                      # gaps de 0.5s

    so_remocao = build_plan(t, TrimConfig(), FPS)
    assert so_remocao.stats.n_cuts == 0

    com_aperto = build_plan(t, TrimConfig(pause_max_seconds=0.2), FPS)
    assert com_aperto.stats.n_squeeze_cuts == 6
    assert com_aperto.stats.removed_seconds == pytest.approx(6 * 0.3, abs=0.05)


def test_aperto_entra_na_chave_de_cache():
    t = fala(("uma", 0.00, 0.30), ("frase", 0.80, 1.20))
    a = build_plan(t, TrimConfig(), FPS)
    b = build_plan(t, TrimConfig(pause_max_seconds=0.2), FPS)
    assert a.input_hash != b.input_hash


def test_estatisticas_fecham_com_aperto():
    rules = TrimConfig(pause_max_seconds=0.2)
    tokens = [(f"p{i}", i * 1.0, i * 1.0 + 0.5) for i in range(5)]
    plan = build_plan(fala(*tokens), rules, FPS)
    s = plan.stats
    assert s.n_cuts == s.n_pause_cuts + s.n_filler_cuts + s.n_squeeze_cuts
    assert s.trimmed_seconds == pytest.approx(sum(r.duration for r in plan.keep), abs=1e-3)


def test_corte_menor_que_um_frame_e_descartado():
    """Abaixo de um frame o corte nao e impreciso, e indeterminado.

    Medido a 30fps: um corte de 10ms remove 0ms ou 33ms dependendo de onde
    cai no grid. Zero e uma emenda de graca no filtergraph — risco de
    artefato sem ganho nenhum; 33ms e o triplo do pretendido.
    """
    from pipeline.trim import representable

    rules = TrimConfig(pause_max_seconds=0.20)
    # gap de 0.21s com teto de 0.20 -> corte de 0.01s, abaixo de 1/30
    t = fala(("uma", 0.00, 0.30), ("frase", 0.51, 1.00))
    assert find_cuts(t, rules)                      # find_cuts ainda o propoe
    assert representable(find_cuts(t, rules), 30.0) == []

    plan = build_plan(t, rules, 30.0)
    assert plan.cuts == []                          # e o plano nao o carrega
    assert len(plan.keep) == 1                      # sem emenda nenhuma


def test_corte_de_um_frame_e_mantido():
    from pipeline.trim import representable

    rules = TrimConfig(pause_max_seconds=0.20)
    t = fala(("uma", 0.00, 0.30), ("frase", 0.54, 1.00))   # corte de 0.04s
    cuts = representable(find_cuts(t, rules), 30.0)
    assert [c.reason for c in cuts] == ["squeeze"]


def test_fps_maior_representa_corte_menor():
    """A 60fps um corte de 0.02s e representavel; a 30 nao e."""
    from pipeline.trim import representable

    rules = TrimConfig(pause_max_seconds=0.20)
    t = fala(("uma", 0.00, 0.30), ("frase", 0.52, 1.00))   # corte de 0.02s
    proposto = find_cuts(t, rules)
    assert representable(proposto, 30.0) == []
    assert len(representable(proposto, 60.0)) == 1
