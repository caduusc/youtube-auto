"""Isolamento por modelo de embedding.

Dois modelos podem ter a mesma dimensao — `all-MiniLM-L6-v2` e
`paraphrase-multilingual-MiniLM-L12-v2` tem 384 os dois — entao a checagem de
tamanho no cosseno nao detecta uma troca de modelo. Sem guardar qual modelo
gerou cada linha, trocar o config produziria similaridades sem sentido e
reusaria imagem errada em silencio.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pipeline.bank import AssetBank, pack


class NamedEmbedder:
    """Dois modelos distintos, MESMA dimensao, espacos incompativeis."""

    def __init__(self, model_name: str, seed: float) -> None:
        self.model_name = model_name
        self.seed = seed

    def embed(self, text: str) -> list[float]:
        vector = [((hash((text, i, self.seed)) % 1000) / 1000.0) for i in range(8)]
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


ENGLISH = "sentence-transformers/all-MiniLM-L6-v2"
MULTI = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def open_bank(tmp_path, model: str, seed: float = 1.0) -> AssetBank:
    return AssetBank(
        root=tmp_path, db_path=tmp_path / "bank.db", images_dir=tmp_path / "images",
        embedder=NamedEmbedder(model, seed), threshold=0.82,
    )


def stored_image(tmp_path, name: str) -> str:
    (tmp_path / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / name).write_bytes(b"x")
    return f"images/{name}"


# --------------------------------------------------------------------------


def test_o_modelo_fica_gravado_na_linha(tmp_path):
    bank = open_bank(tmp_path, ENGLISH)
    bank.add(path=stored_image(tmp_path, "a.png"), concept="a desk", origin="generated")
    row = bank.conn.execute("SELECT embedding_model FROM assets").fetchone()
    assert row["embedding_model"] == ENGLISH


def test_trocar_de_modelo_nao_reusa_o_banco_antigo(tmp_path):
    """O ponto todo: mesma dimensao, espacos diferentes, zero reuso."""
    old = open_bank(tmp_path, MULTI, seed=1.0)
    old.add(path=stored_image(tmp_path, "desk.png"), concept="a desk with a notebook",
            origin="generated", cost_usd=0.025)
    old.close()

    new = open_bank(tmp_path, ENGLISH, seed=2.0)
    assert new.find_similar("a desk with a notebook") is None


def test_o_mesmo_modelo_continua_reusando(tmp_path):
    first = open_bank(tmp_path, ENGLISH)
    first.add(path=stored_image(tmp_path, "desk.png"), concept="a desk with a notebook",
              origin="generated")
    first.close()

    again = open_bank(tmp_path, ENGLISH)
    hit = again.find_similar("a desk with a notebook")
    assert hit is not None and hit[1] == pytest.approx(1.0)


def test_stats_mostra_as_duas_familias(tmp_path):
    old = open_bank(tmp_path, MULTI, seed=1.0)
    old.add(path=stored_image(tmp_path, "a.png"), concept="a desk", origin="generated")
    old.close()

    new = open_bank(tmp_path, ENGLISH, seed=2.0)
    new.add(path=stored_image(tmp_path, "b.png"), concept="a city", origin="stock")

    stats = new.stats()
    assert stats["active_model"] == ENGLISH
    assert stats["by_model"] == {MULTI: 1, ENGLISH: 1}
    assert stats["assets"] == 2      # as duas contam para custo e prune


def test_curto_circuito_considera_o_modelo(tmp_path):
    """Banco cheio de OUTRO modelo equivale a banco vazio para este: nao
    pode nem carregar o embedder."""
    old = open_bank(tmp_path, MULTI, seed=1.0)
    old.add(path=stored_image(tmp_path, "a.png"), concept="a desk", origin="generated")
    old.close()

    class Exploding(NamedEmbedder):
        def embed(self, text: str):
            raise AssertionError("o embedder foi carregado sem nada comparavel no banco")

    new = open_bank(tmp_path, ENGLISH, seed=2.0)
    new.embedder = Exploding(ENGLISH, 2.0)
    assert new.find_similar("a desk") is None


# --------------------------------------------------------------------------
# migracao de banco antigo
# --------------------------------------------------------------------------


def test_banco_sem_a_coluna_e_migrado(tmp_path):
    """Quem ja tinha assets nao pode ver o banco inteiro orfanado."""
    import sqlite3

    db = tmp_path / "bank.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE, concept TEXT NOT NULL, prompt TEXT,
            origin TEXT NOT NULL CHECK (origin IN ('stock','generated')),
            embedding BLOB NOT NULL, embedding_dim INTEGER NOT NULL,
            cost_usd REAL NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
            use_count INTEGER NOT NULL DEFAULT 0, last_used_at TEXT
        );
    """)
    vector = NamedEmbedder(ENGLISH, 1.0).embed("a desk with a notebook")
    conn.execute(
        "INSERT INTO assets (path, concept, origin, embedding, embedding_dim, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (stored_image(tmp_path, "legacy.png"), "a desk with a notebook", "generated",
         pack(vector), len(vector), "2026-01-01T00:00:00Z"),
    )
    conn.commit()
    conn.close()

    bank = open_bank(tmp_path, ENGLISH, seed=1.0)
    # a linha legada ganhou o modelo ativo, que e o ponto deste teste
    assert bank.stats()["by_model"] == {ENGLISH: 1}
    # Ela NAO volta pelo reuso, e a razao e a outra migracao: sem `prompt`
    # guardado nao ha como saber em que estilo foi gerada, e reusar imagem de
    # estilo desconhecido e o dano que a coluna `style` existe para evitar.
    # O caso recuperavel esta em tests/test_style_isolation.py.
    assert bank.stats()["by_style"] == {"?": 1}
    assert bank.find_similar("a desk with a notebook") is None


# --------------------------------------------------------------------------
# pasta local vs ID do HuggingFace
# --------------------------------------------------------------------------


def test_id_do_hub_passa_inalterado(tmp_path):
    from pipeline.embed import resolve_model

    identity, load_path = resolve_model(ENGLISH, tmp_path)
    assert identity == ENGLISH
    assert load_path == ENGLISH


def test_pasta_local_e_resolvida_contra_a_raiz(tmp_path):
    """Sem isso, o pipeline rodado de outro diretorio nao acha o modelo."""
    from pipeline.embed import resolve_model

    (tmp_path / "models" / "all-MiniLM-L6-v2").mkdir(parents=True)
    identity, load_path = resolve_model("models/all-MiniLM-L6-v2", tmp_path)

    assert Path(load_path).is_absolute()
    assert Path(load_path) == (tmp_path / "models" / "all-MiniLM-L6-v2").resolve()


def test_identidade_nao_vira_caminho_absoluto(tmp_path):
    """A identidade vai para cada linha do banco. Se fosse o caminho
    absoluto, mover o projeto invalidaria o banco inteiro."""
    from pipeline.embed import resolve_model

    (tmp_path / "models" / "all-MiniLM-L6-v2").mkdir(parents=True)
    identity, _ = resolve_model("models/all-MiniLM-L6-v2", tmp_path)
    assert identity == "models/all-MiniLM-L6-v2"


def test_pasta_inexistente_cai_para_id_do_hub(tmp_path):
    from pipeline.embed import resolve_model

    identity, load_path = resolve_model("models/nao-existe", tmp_path)
    assert identity == load_path == "models/nao-existe"


def test_embedder_usa_load_path_e_reporta_model_name(tmp_path):
    from pipeline.embed import SentenceTransformerEmbedder

    e = SentenceTransformerEmbedder("models/local", str(tmp_path / "models" / "local"))
    assert e.model_name == "models/local"
    assert e.load_path == str(tmp_path / "models" / "local")
