"""O que acontece quando indexar falha depois de baixar ou gerar.

Caso real: o modelo de embedding falhou no download (erro de transferencia do
HuggingFace) DEPOIS de a imagem de stock ja estar em disco. Numa imagem
gerada, o mesmo caminho perderia dinheiro ja gasto.
"""

from __future__ import annotations

import pytest
from conftest import span
from test_bank_reuse import FakeEmbedder, FakeProvider, FakeStock, make_resolver

from pipeline.bank import AssetBank
from pipeline.edl import resolve as resolve_edl
from pipeline.schemas import PlannedEDL


class BrokenEmbedder:
    """Falha como o HuggingFace falhou: no carregamento do modelo."""

    def embed(self, text: str) -> list[float]:
        raise OSError("Can't load the model for 'paraphrase-multilingual-MiniLM-L12-v2'")


@pytest.fixture
def healthy_bank(tmp_path):
    return AssetBank(
        root=tmp_path, db_path=tmp_path / "assets" / "bank.db",
        images_dir=tmp_path / "assets" / "images",
        embedder=FakeEmbedder(), threshold=0.82,
    )


@pytest.fixture
def broken_bank(tmp_path):
    return AssetBank(
        root=tmp_path, db_path=tmp_path / "assets" / "bank.db",
        images_dir=tmp_path / "assets" / "images",
        embedder=BrokenEmbedder(), threshold=0.82,
    )


def one_broll(transcript, concept: str, tags: list[str]):
    s = span("broll", 6, 8)
    s.concept, s.concept_tags = concept, tags
    spans = [span("aroll", 0, 5), s, span("aroll", 9, 199)]
    return resolve_edl(PlannedEDL(spans=spans), transcript)


# --------------------------------------------------------------------------
# preflight: falhar antes de gastar
# --------------------------------------------------------------------------


def test_warmup_falha_antes_de_gastar(broken_bank):
    with pytest.raises(RuntimeError, match="nao foi possivel carregar o modelo de embedding"):
        broken_bank.warmup()


def test_mensagem_do_warmup_diz_que_nada_foi_gasto(broken_bank):
    with pytest.raises(RuntimeError) as excinfo:
        broken_bank.warmup()
    message = str(excinfo.value)
    assert "Nada foi gasto" in message
    assert "HF_HUB_DISABLE_XET" in message   # o contorno no Windows


class TLSBrokenEmbedder:
    """Falha como a maquina real falhou: registros TLS corrompidos."""

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def embed(self, text: str) -> list[float]:
        raise OSError(
            "[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed "
            "or bad record mac (_ssl.c:2580)"
        )


def test_erro_de_tls_nao_recomenda_tentar_de_novo(broken_bank):
    """TLS corrompido nao e transitorio: arquivo pequeno passa e
    transferencia sustentada quebra. Mandar reexecutar e conselho ruim."""
    broken_bank.embedder = TLSBrokenEmbedder()
    with pytest.raises(RuntimeError) as excinfo:
        broken_bank.warmup()
    message = str(excinfo.value)

    assert "TLS corrompido" in message
    assert "Reexecutar nao" in message and "resolve" in message
    assert "hf_transfer" in message        # outra pilha TLS
    assert "antivirus" in message          # a causa mais comum
    assert "pasta local" in message        # o escape garantido
    assert "costuma ser transitorio" not in message


def test_mensagem_nomeia_o_modelo_que_falhou(broken_bank):
    broken_bank.embedder = TLSBrokenEmbedder()
    with pytest.raises(RuntimeError, match="all-MiniLM-L6-v2"):
        broken_bank.warmup()


def test_warmup_passa_com_embedder_sadio(healthy_bank):
    healthy_bank.warmup()   # nao levanta


# --------------------------------------------------------------------------
# rede de seguranca: indexar falhando nao descarta o asset
# --------------------------------------------------------------------------


def test_imagem_gerada_nao_e_perdida_se_indexar_falhar(broken_bank, tmp_path, transcript):
    """O dinheiro ja saiu. O asset tem que sobreviver."""
    provider = FakeProvider(cost=0.025)
    segments = one_broll(transcript, "a desk with a notebook", ["desk", "notebook"])
    assets = make_resolver(broken_bank, tmp_path, provider=provider).resolve(segments)

    item = assets.items[0]
    assert item.origin == "generated"
    assert item.path is not None                      # o render ainda usa a imagem
    assert (tmp_path / item.path).exists()            # o arquivo esta em disco
    assert item.cost_usd == pytest.approx(0.025)      # o custo e contabilizado
    assert "sem embedding" in item.note               # e o motivo fica registrado
    assert broken_bank.stats()["assets"] == 1         # a linha existe no banco
    assert broken_bank.stats()["total_spent_usd"] == pytest.approx(0.025)


def test_imagem_de_stock_nao_e_perdida_se_indexar_falhar(broken_bank, tmp_path, transcript):
    stock = FakeStock(["desk"])
    segments = one_broll(transcript, "a desk with a notebook", ["desk", "notebook"])
    assets = make_resolver(broken_bank, tmp_path, stock=stock).resolve(segments)

    item = assets.items[0]
    assert item.origin == "stock"
    assert item.path is not None and (tmp_path / item.path).exists()
    assert broken_bank.stats()["assets"] == 1


def test_asset_sem_embedding_nunca_e_reusado(broken_bank, tmp_path, transcript):
    """Embedding vazio nao pode virar um falso positivo de similaridade."""
    provider = FakeProvider()
    segments = one_broll(transcript, "a desk with a notebook", ["desk", "notebook"])
    make_resolver(broken_bank, tmp_path, provider=provider).resolve(segments)

    # agora com um embedder sadio, o asset degradado nao deve casar com nada
    broken_bank.embedder = FakeEmbedder()
    assert broken_bank.find_similar("a notebook on a desk") is None


def test_backfill_manual_recupera_o_asset(broken_bank, tmp_path, transcript):
    """A linha degradada nao e lixo permanente: reindexar a mao volta a
    valer, o que importa porque o asset ja foi pago."""
    provider = FakeProvider()
    segments = one_broll(transcript, "a desk with a notebook", ["desk", "notebook"])
    make_resolver(broken_bank, tmp_path, provider=provider).resolve(segments)

    broken_bank.embedder = FakeEmbedder()
    row = broken_bank.conn.execute("SELECT id, concept FROM assets").fetchone()
    from pipeline.bank import pack
    broken_bank.conn.execute(
        "UPDATE assets SET embedding = ?, embedding_dim = ? WHERE id = ?",
        (pack(broken_bank.embedder.embed(row["concept"])), 8, row["id"]),
    )
    broken_bank.conn.commit()

    hit = broken_bank.find_similar("a notebook on a desk")
    assert hit is not None and hit[1] >= 0.82


# --------------------------------------------------------------------------
# dry-run nao paga o preco do preflight
# --------------------------------------------------------------------------


def test_dry_run_nao_precisa_do_embedder_com_banco_vazio(broken_bank, tmp_path, transcript):
    """O --dry-run existe para responder na hora; com banco vazio ele nao tem
    o que comparar e nao pode exigir 400MB de modelo."""
    segments = one_broll(transcript, "a desk with a notebook", ["desk", "notebook"])
    assets = make_resolver(broken_bank, tmp_path).resolve(segments, dry_run=True)
    assert assets.dry_run is True
    assert assets.estimate.n_to_generate == 1


# --------------------------------------------------------------------------
# retries: download insiste mais que chamada que cobra
# --------------------------------------------------------------------------


def test_download_insiste_mais_que_a_api(config):
    """Baixar por URL e idempotente e de graca, entao pode insistir. Criar
    predicao cobra: se ela roda no servidor e a resposta se perde, repetir
    gera uma segunda imagem e cobra duas vezes."""
    assert config.network.download_attempts > config.network.api_attempts
    assert config.network.api_attempts == 3      # o que o spec pede


def test_download_grava_em_temporario_e_move_no_fim(tmp_path, monkeypatch, config):
    """Escrever direto no destino deixaria arquivo truncado no lugar se a
    conexao caisse no meio, e o resto do pipeline trataria como valido."""
    import pipeline.providers as providers

    destino = tmp_path / "img.png"
    vistos = []

    class Response:
        def raise_for_status(self): pass
        def iter_content(self, chunk_size):
            # no meio do stream, o destino final ainda nao pode existir
            vistos.append(destino.exists())
            yield b"\x89PNG"
            vistos.append(destino.exists())
            yield b"-resto"

    monkeypatch.setattr(providers.requests, "get", lambda *a, **k: Response())
    providers._download(
        "https://example.test/x.png", destino, timeout=5, network=config.network)

    assert vistos == [False, False]          # nao apareceu antes de completar
    assert destino.read_bytes() == b"\x89PNG-resto"
    assert not destino.with_suffix(".png.part").exists()


def test_temporario_e_limpo_quando_o_download_falha(tmp_path, monkeypatch, config):
    import pipeline.providers as providers

    destino = tmp_path / "img.png"

    def sempre_falha(*a, **k):
        raise OSError("[SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] bad record mac")

    monkeypatch.setattr(providers.requests, "get", sempre_falha)
    net = config.network.model_copy()
    net.download_attempts = 2
    net.base_delay_seconds = 0.0

    with pytest.raises(RuntimeError):
        providers._download("https://example.test/x.png", destino, timeout=5, network=net)

    assert not destino.exists()
    assert not destino.with_suffix(".png.part").exists()
