"""O formato do roteiro em disco.

Este e o unico artefato do pipeline que existe para ser EDITADO A MAO, e e o
que faz o round-trip importar: o estagio 1 escreve, voce edita, o estagio 2 le.
Se ida e volta nao fossem iguais, uma edicao sua poderia ser desfeita em
silencio.
"""

from __future__ import annotations

import pytest

from pipeline.script import (
    ScriptFormatError,
    parse,
    parse_beats,
    read,
    render,
    spoken_seconds,
    words,
    write,
)
from pipeline.schemas import Script, ScriptBeat

VALIDO = """\
---
slug: video-de-teste
subject: como montar uma esteira de producao de video
argument: a automacao ja e barata o bastante
viewer_takeaway: da para montar o estudio em software
duration_target_seconds: 900.0
---

## beat 1 — abertura
Primeira frase do roteiro.
Segunda linha do mesmo beat.

## beat 2 — o problema
Texto do segundo beat.
"""


def test_parse_do_formato_valido():
    s = parse(VALIDO)
    assert s.slug == "video-de-teste"
    assert s.duration_target_seconds == 900.0
    assert [b.id for b in s.beats] == [1, 2]
    assert s.beats[0].title == "abertura"
    assert s.beats[0].text == "Primeira frase do roteiro.\nSegunda linha do mesmo beat."


def test_round_trip_preserva_tudo():
    """`parse(render(s)) == s`. E o que protege a sua edicao."""
    original = parse(VALIDO)
    assert parse(render(original)) == original


def test_round_trip_com_acento_e_travessao():
    s = Script(
        slug="x", subject="ação e reação", argument="é só isso",
        viewer_takeaway="não há mais", duration_target_seconds=120.0,
        beats=[ScriptBeat(id=1, title="atenção — parte 1", text="çãé íóú")],
    )
    assert parse(render(s)) == s


# --------------------------------------------------------------------------
# o id vem do cabecalho, nao da posicao
# --------------------------------------------------------------------------


def test_id_vem_do_cabecalho_nao_da_ordem():
    """Apagar um beat do meio nao pode renumerar os outros.

    O storyboard referencia por id; se o id viesse da posicao, apagar o beat 2
    faria o 3 virar 2 e a imagem do 3 passar a ilustrar outro trecho.
    """
    s = parse(VALIDO.replace("## beat 2 —", "## beat 7 —"))
    assert [b.id for b in s.beats] == [1, 7]


def test_id_duplicado_e_rejeitado():
    with pytest.raises(ScriptFormatError, match="duas vezes"):
        parse(VALIDO.replace("## beat 2 —", "## beat 1 —"))


@pytest.mark.parametrize("cabecalho", [
    "## beat 3 — titulo",
    "## beat 3 - titulo",
    "## beat 3 – titulo",
    "## beat 3",
    "##  BEAT  3  —  titulo",
])
def test_variantes_de_cabecalho(cabecalho):
    """Quem edita a mao digita o que o teclado tem."""
    beats = parse_beats(f"{cabecalho}\ntexto\n")
    assert [b.id for b in beats] == [3]


# --------------------------------------------------------------------------
# erros que sao recado para uma pessoa
# --------------------------------------------------------------------------


def test_sem_frontmatter():
    with pytest.raises(ScriptFormatError, match="frontmatter"):
        parse("## beat 1\ntexto\n")


def test_frontmatter_incompleto_lista_o_que_falta():
    ruim = "---\nslug: x\nsubject: y\n---\n\n## beat 1\ntexto\n"
    with pytest.raises(ScriptFormatError, match="argument.*viewer_takeaway"):
        parse(ruim)


def test_frontmatter_que_nao_e_yaml():
    with pytest.raises(ScriptFormatError, match="YAML"):
        parse("---\n: : :\n  - x\n---\n\n## beat 1\ntexto\n")


def test_roteiro_sem_beat():
    with pytest.raises(ScriptFormatError, match="beat nenhum"):
        parse(VALIDO[:VALIDO.index("## beat 1")])


def test_beat_vazio_aponta_o_numero():
    with pytest.raises(ScriptFormatError, match="beat 2 esta vazio"):
        parse(VALIDO.replace("Texto do segundo beat.", "   "))


def test_texto_antes_do_primeiro_beat_e_ignorado():
    """E onde uma nota para voce mesmo cabe sem virar conteudo do video."""
    com_nota = VALIDO.replace("## beat 1", "Nota: gravar de manha.\n\n## beat 1", 1)
    assert parse(com_nota) == parse(VALIDO)


# --------------------------------------------------------------------------
# disco
# --------------------------------------------------------------------------


def test_write_e_read(tmp_path):
    caminho = tmp_path / "sub" / "script.md"
    original = parse(VALIDO)
    write(caminho, original)
    assert read(caminho) == original


def test_read_de_arquivo_ausente_nao_vaza_oserror(tmp_path):
    with pytest.raises(ScriptFormatError, match="nao encontrado"):
        read(tmp_path / "nao-existe.md")


def test_erro_de_leitura_nomeia_o_arquivo(tmp_path):
    caminho = tmp_path / "script.md"
    caminho.write_text("sem frontmatter", encoding="utf-8")
    with pytest.raises(ScriptFormatError, match="script.md"):
        read(caminho)


# --------------------------------------------------------------------------
# duracao estimada
# --------------------------------------------------------------------------


def test_contagem_de_palavras_ignora_frontmatter():
    s = parse(VALIDO)
    assert words(s) == len("Primeira frase do roteiro. Segunda linha do mesmo beat.".split()) \
                       + len("Texto do segundo beat.".split())


def test_duracao_falada():
    """150 palavras por minuto -> 150 palavras dao 60s."""
    s = Script(slug="x", subject="a", argument="b", viewer_takeaway="c",
               beats=[ScriptBeat(id=1, text=" ".join(["palavra"] * 150))])
    assert spoken_seconds(s, 150.0) == pytest.approx(60.0)


def test_duracao_falada_nao_divide_por_zero():
    s = Script(slug="x", subject="a", argument="b", viewer_takeaway="c",
               beats=[ScriptBeat(id=1, text="uma palavra")])
    assert spoken_seconds(s, 0.0) > 0
