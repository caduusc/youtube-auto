"""Isolamento por estilo.

`style_suffix` existe para 20 imagens geradas em 20 chamadas independentes
parecerem sair da mao do mesmo ilustrador. O embedding do banco e do
`concept`, que nao carrega estilo nenhum — entao sem guardar o estilo de cada
linha, trocar de estilo faria o banco reusar a imagem antiga no conceito que
casasse, desfazendo em silencio exatamente aquilo.

Mesma familia do isolamento por modelo de embedding, com uma diferenca na
migracao: o modelo antigo pode ser assumido, o estilo antigo nao — quem migra
esta provavelmente estreando um estilo novo. O que salva o reuso e o `prompt`
guardado, que contem o sufixo de estilo de fato enviado ao provider.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pipeline.bank import AssetBank

WARM = "warm sand editorial illustration, flat vector"
DARK = "dark old money illustration, deep green and brass"


class FixedEmbedder:
    model_name = "test-model"

    def embed(self, text: str) -> list[float]:
        vector = [((hash((text, i)) % 1000) / 1000.0) for i in range(8)]
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


def open_bank(tmp_path, style: str, suffix: str = "", **kw) -> AssetBank:
    styles = kw.pop("styles", None)
    if styles is None:
        styles = {style: suffix} if suffix else {}
    return AssetBank(
        root=tmp_path, db_path=tmp_path / "bank.db", images_dir=tmp_path / "images",
        embedder=FixedEmbedder(), threshold=0.82, style=style, styles=styles,
    )


def stored_image(tmp_path, name: str) -> str:
    (tmp_path / "images").mkdir(parents=True, exist_ok=True)
    path = tmp_path / "images" / name
    path.write_bytes(b"\x89PNG")
    return str(Path("images") / name)


CONCEITO = "a wooden desk with scattered papers"


# --------------------------------------------------------------------------
# o reuso e limitado ao estilo ativo
# --------------------------------------------------------------------------


def test_imagem_do_mesmo_estilo_e_reusada(tmp_path):
    bank = open_bank(tmp_path, "warm")
    bank.add(path=stored_image(tmp_path, "a.png"), concept=CONCEITO,
             origin="generated", prompt=f"{CONCEITO}, {WARM}")
    bank.close()

    assert open_bank(tmp_path, "warm").find_similar(CONCEITO) is not None


def test_imagem_de_outro_estilo_nao_e_reusada(tmp_path):
    """O caso que motivou a coluna: o concept casa, o estilo nao.

    Sem o filtro, trocar `style` no config daria reuso da imagem antiga no
    video novo — e o espectador ve dois ilustradores no mesmo canal.
    """
    bank = open_bank(tmp_path, "warm")
    bank.add(path=stored_image(tmp_path, "a.png"), concept=CONCEITO,
             origin="generated", prompt=f"{CONCEITO}, {WARM}")
    bank.close()

    assert open_bank(tmp_path, "old_money").find_similar(CONCEITO) is None


def test_stock_vale_para_qualquer_estilo(tmp_path):
    """Foto nao tem estilo de ilustracao para casar ou destoar.

    E o oposto do teste acima de proposito: limitar o stock por estilo faria
    o pipeline pagar geracao por imagem generica que ja tem de graca.
    """
    bank = open_bank(tmp_path, "warm")
    bank.add(path=stored_image(tmp_path, "s.png"), concept=CONCEITO, origin="stock")
    bank.close()

    assert open_bank(tmp_path, "old_money").find_similar(CONCEITO) is not None


def test_stats_separa_por_estilo(tmp_path):
    bank = open_bank(tmp_path, "warm")
    bank.add(path=stored_image(tmp_path, "a.png"), concept=CONCEITO,
             origin="generated", prompt="x")
    bank.add(path=stored_image(tmp_path, "s.png"), concept="outra coisa", origin="stock")
    bank.close()

    stats = open_bank(tmp_path, "old_money").stats()
    assert stats["active_style"] == "old_money"
    assert stats["by_style"] == {"warm": 1}      # so gerada; stock nao entra


# --------------------------------------------------------------------------
# migracao de banco sem a coluna
# --------------------------------------------------------------------------


def legacy_bank(tmp_path, prompt: str | None) -> None:
    """Banco no formato anterior a coluna `style`."""
    import sqlite3

    from pipeline.bank import pack

    conn = sqlite3.connect(tmp_path / "bank.db")
    conn.executescript("""
        CREATE TABLE assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE, concept TEXT NOT NULL, prompt TEXT,
            origin TEXT NOT NULL CHECK (origin IN ('stock','generated')),
            embedding BLOB NOT NULL, embedding_dim INTEGER NOT NULL,
            embedding_model TEXT NOT NULL DEFAULT '',
            cost_usd REAL NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
            use_count INTEGER NOT NULL DEFAULT 0, last_used_at TEXT
        );
    """)
    vector = FixedEmbedder().embed(CONCEITO)
    conn.execute(
        "INSERT INTO assets (path, concept, prompt, origin, embedding, embedding_dim,"
        " embedding_model, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (stored_image(tmp_path, "legacy.png"), CONCEITO, prompt, "generated",
         pack(vector), len(vector), "test-model", "2026-01-01T00:00:00Z"),
    )
    conn.commit()
    conn.close()


def test_migracao_reconhece_o_estilo_pelo_prompt(tmp_path):
    """O reuso e requisito de custo, nao otimizacao: orfanar tudo sai caro.

    Da para ser exato em vez de chutar. `prompt` guarda `concept + sufixo`,
    entao a linha cujo prompt termina com o estilo ativo E daquele estilo —
    lido do que de fato foi enviado ao provider, nao assumido.
    """
    legacy_bank(tmp_path, prompt=f"{CONCEITO}, {WARM}")
    bank = open_bank(tmp_path, "warm", suffix=WARM)

    assert bank.stats()["by_style"] == {"warm": 1}
    assert bank.find_similar(CONCEITO) is not None


def test_migracao_orfana_quando_o_prompt_e_de_outro_estilo(tmp_path):
    legacy_bank(tmp_path, prompt=f"{CONCEITO}, {WARM}")
    bank = open_bank(tmp_path, "old_money", suffix=DARK)

    assert bank.stats()["by_style"] == {"?": 1}
    assert bank.find_similar(CONCEITO) is None


def test_migracao_orfana_quando_nao_ha_prompt(tmp_path):
    """Sem prompt nao ha o que ler, e assumir o estilo ativo seria a mentira
    que a coluna existe para impedir."""
    legacy_bank(tmp_path, prompt=None)
    bank = open_bank(tmp_path, "old_money", suffix=DARK)

    assert bank.stats()["by_style"] == {"?": 1}
    assert bank.find_similar(CONCEITO) is None


def test_migracao_nao_repete_em_abertura_seguinte(tmp_path):
    legacy_bank(tmp_path, prompt=f"{CONCEITO}, {WARM}")
    open_bank(tmp_path, "warm", suffix=WARM).close()
    # segunda abertura: a coluna ja existe, nada e remarcado
    assert open_bank(tmp_path, "warm", suffix=WARM).stats()["by_style"] == {"warm": 1}


def test_migracao_reconhece_estilo_nao_ativo(tmp_path):
    """O mapa inteiro, nao so o estilo ativo.

    Quem mantem dois estilos no config e troca entre eles nao perde o reuso
    do que nao esta ativo agora: a imagem warm fica marcada como warm, e
    volta a ser reusavel quando ele voltar para aquele estilo.
    """
    legacy_bank(tmp_path, prompt=f"{CONCEITO}, {WARM}")
    conhecidos = {"warm": WARM, "old_money": DARK}

    bank = open_bank(tmp_path, "old_money", styles=conhecidos)
    assert bank.stats()["by_style"] == {"warm": 1}   # reconhecida, nao orfa
    assert bank.find_similar(CONCEITO) is None        # e nao reusada agora
    bank.close()

    # de volta ao estilo warm, a imagem volta a servir
    assert open_bank(tmp_path, "warm", styles=conhecidos).find_similar(CONCEITO) is not None



# --------------------------------------------------------------------------
# resolucao do estilo no config
# --------------------------------------------------------------------------


def base_config(**kw) -> dict:
    return dict(
        anthropic=dict(model="m", env="E"),
        editorial=dict(broll_min_seconds=3, broll_max_seconds=9,
                       max_switches_per_minute=10, intro_aroll_seconds=8,
                       broll_ratio_min=0.35, broll_ratio_max=0.70),
        bank=dict(db_path="a", images_dir="b", embedding_model="c",
                  similarity_threshold=0.82),
        budget=dict(max_usd_per_video=1.5, brl_per_usd=5.4),
        image_provider=dict(active="none",
                            replicate=dict(env="R", model="m", cost_usd_per_image=0.003)),
        stock=dict(env="P"),
        **kw,
    )


def test_style_nomeado_resolve_para_o_texto():
    from pipeline.config import Config

    c = Config(**base_config(style="old_money", styles={"old_money": DARK, "warm": WARM}))
    assert c.style_suffix == DARK
    assert c.style_name == "old_money"
    assert c.known_styles == {"old_money": DARK, "warm": WARM}


def test_style_suffix_literal_continua_valendo():
    """Config antigo nao quebra: sem `styles`, o texto direto ainda serve."""
    from pipeline.config import Config

    c = Config(**base_config(style_suffix=WARM))
    assert c.style_suffix == WARM
    assert c.style_name.startswith("suffix:")
    assert c.known_styles == {c.style_name: WARM}


def test_identidade_do_literal_acompanha_o_texto():
    """Dois textos diferentes tem identidades diferentes, e o mesmo texto tem
    sempre a mesma — senao o banco perderia o reuso a cada abertura."""
    from pipeline.config import Config

    a = Config(**base_config(style_suffix=WARM)).style_name
    b = Config(**base_config(style_suffix=WARM + " ")).style_name
    d = Config(**base_config(style_suffix=DARK)).style_name
    assert a == b          # espaco na ponta nao muda o estilo
    assert a != d


@pytest.mark.parametrize("kw,esperado", [
    (dict(style="fantasma", styles={"a": "x"}), "nao existe em"),
    (dict(style="a", styles={"a": "x"}, style_suffix="y"), "nao os dois"),
    (dict(), "nenhum estilo definido"),
    (dict(style_suffix="   "), "nenhum estilo definido"),
])
def test_estilo_invalido_para_antes_de_gastar(kw, esperado):
    """Estilo errado nao quebra nada: produz um video com as imagens erradas,
    e ai a geracao ja foi paga. Tem que parar na validacao do config."""
    import pydantic

    from pipeline.config import Config

    with pytest.raises(pydantic.ValidationError, match=esperado):
        Config(**base_config(**kw))
