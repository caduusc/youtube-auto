"""Reuso do banco de assets. Nenhuma chamada externa, nenhum custo real."""

from __future__ import annotations

import math

import pytest
from conftest import span

from pipeline.bank import AssetBank, cosine
from pipeline.edl import resolve as resolve_edl
from pipeline.resolver import AssetResolver, BudgetExceeded
from pipeline.schemas import PlannedEDL

# --------------------------------------------------------------------------
# dublês
# --------------------------------------------------------------------------


class FakeEmbedder:
    """Embedding deterministico por palavra: conceitos que compartilham
    palavras ficam proximos, conceitos disjuntos ficam ortogonais."""

    VOCAB = ["desk", "notebook", "city", "ocean", "kitchen", "server", "forest", "bridge"]

    def embed(self, text: str) -> list[float]:
        words = text.lower().split()
        vector = [float(sum(1 for w in words if term in w)) for term in self.VOCAB]
        if not any(vector):
            vector = [1.0] + [0.0] * (len(self.VOCAB) - 1)
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]


class FakeProvider:
    def __init__(self, cost: float = 0.025, actual_cost: float | None = None) -> None:
        self._cost = cost                                  # o que o config anuncia
        self._actual = cost if actual_cost is None else actual_cost  # o que cobra
        self.calls: list[str] = []

    @property
    def cost_usd_per_image(self) -> float:
        return self._cost

    def generate(self, prompt: str, destination) -> float:
        self.calls.append(prompt)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x89PNG-fake")
        return self._actual


class FakeStock:
    def __init__(self, generic_tags: list[str], *, hit: bool = True) -> None:
        self.generic_tags = generic_tags
        self.hit = hit
        self.searches: list[list[str]] = []

    def is_generic(self, tags: list[str]) -> bool:
        haystack = " ".join(tags).lower()
        return any(g.lower() in haystack for g in self.generic_tags)

    def search(self, tags: list[str]):
        self.searches.append(list(tags))
        return ("https://example.test/photo.jpg", "Pexels / Someone") if self.hit else None

    def download(self, url: str, destination) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\xff\xd8-fake")


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def bank(tmp_path):
    return AssetBank(
        root=tmp_path,
        db_path=tmp_path / "assets" / "bank.db",
        images_dir=tmp_path / "assets" / "images",
        embedder=FakeEmbedder(),
        threshold=0.82,
    )


def make_resolver(bank, tmp_path, *, provider=None, stock=None, budget=1.50, unit=None):
    provider = provider if provider is not None else FakeProvider()
    return AssetResolver(
        bank=bank,
        provider=provider,
        stock=stock,
        style_suffix="flat editorial illustration, muted palette",
        budget_usd=budget,
        cost_usd_per_image=unit if unit is not None else provider.cost_usd_per_image,
        images_dir=tmp_path / "assets" / "images",
    )


def broll(transcript, concepts: list[tuple[str, list[str]]]):
    """Constroi segmentos de b-roll com os conceitos dados."""
    spans, cursor = [span("aroll", 0, 5)], 6
    for concept, tags in concepts:
        s = span("broll", cursor, cursor + 2)
        s.concept, s.concept_tags = concept, tags
        spans.append(s)
        cursor += 3
        spans.append(span("aroll", cursor, cursor + 1))
        cursor += 2
    spans[-1] = span("aroll", spans[-1].first_segment, 199)
    return resolve_edl(PlannedEDL(spans=spans), transcript)


# --------------------------------------------------------------------------
# banco vazio
# --------------------------------------------------------------------------


def test_banco_vazio_manda_tudo_para_geracao(bank, tmp_path, transcript):
    provider = FakeProvider()
    segments = broll(transcript, [("a desk with a notebook", ["desk", "notebook"]),
                                  ("an ocean at dawn", ["ocean", "dawn"])])
    assets = make_resolver(bank, tmp_path, provider=provider).resolve(segments)

    assert assets.by_origin == {"generated": 2}
    assert len(provider.calls) == 2
    assert assets.total_cost_usd == pytest.approx(0.05)


def test_style_suffix_entra_no_prompt(bank, tmp_path, transcript):
    provider = FakeProvider()
    segments = broll(transcript, [("a desk with a notebook", ["desk", "notebook"])])
    make_resolver(bank, tmp_path, provider=provider).resolve(segments)

    assert provider.calls[0].startswith("a desk with a notebook,")
    assert "flat editorial illustration" in provider.calls[0]


# --------------------------------------------------------------------------
# reuso
# --------------------------------------------------------------------------


def test_reusa_do_banco_em_vez_de_gerar(bank, tmp_path, transcript):
    bank.add(path="assets/images/desk.png", concept="a desk with a notebook",
             origin="generated", cost_usd=0.025)
    (tmp_path / "assets" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "assets" / "images" / "desk.png").write_bytes(b"x")

    provider = FakeProvider()
    segments = broll(transcript, [("a notebook on a desk", ["desk", "notebook"])])
    assets = make_resolver(bank, tmp_path, provider=provider).resolve(segments)

    assert assets.by_origin == {"bank": 1}
    assert provider.calls == []
    assert assets.total_cost_usd == 0.0
    assert assets.items[0].similarity >= 0.82


def test_abaixo_do_limiar_nao_reusa(bank, tmp_path, transcript):
    bank.add(path="assets/images/ocean.png", concept="an ocean at dawn",
             origin="generated", cost_usd=0.025)
    (tmp_path / "assets" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "assets" / "images" / "ocean.png").write_bytes(b"x")

    provider = FakeProvider()
    segments = broll(transcript, [("a desk with a notebook", ["desk", "notebook"])])
    assets = make_resolver(bank, tmp_path, provider=provider).resolve(segments)

    assert assets.by_origin == {"generated": 1}
    assert len(provider.calls) == 1


def test_nao_reusa_a_mesma_imagem_duas_vezes_no_mesmo_video(bank, tmp_path, transcript):
    """Reusar entre videos e o ponto; reusar dentro do mesmo video o
    espectador nota, entao o segundo segmento tem que gerar."""
    bank.add(path="assets/images/desk.png", concept="a desk with a notebook",
             origin="generated", cost_usd=0.025)
    (tmp_path / "assets" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "assets" / "images" / "desk.png").write_bytes(b"x")

    provider = FakeProvider()
    segments = broll(transcript, [("a notebook on a desk", ["desk", "notebook"]),
                                  ("a desk and a notebook", ["desk", "notebook"])])
    assets = make_resolver(bank, tmp_path, provider=provider).resolve(segments)

    assert assets.by_origin == {"bank": 1, "generated": 1}
    assert len(provider.calls) == 1


def test_imagem_gerada_volta_para_o_banco(bank, tmp_path, transcript):
    segments = broll(transcript, [("a desk with a notebook", ["desk", "notebook"])])
    make_resolver(bank, tmp_path).resolve(segments)

    assert bank.stats()["assets"] == 1
    assert bank.stats()["by_origin"] == {"generated": 1}
    assert bank.stats()["total_uses"] == 1


def test_arquivo_sumido_nao_e_reusado(bank, tmp_path, transcript):
    """Linha no banco sem arquivo no disco nao vale como hit."""
    bank.add(path="assets/images/gone.png", concept="a desk with a notebook",
             origin="generated", cost_usd=0.025)
    provider = FakeProvider()
    segments = broll(transcript, [("a notebook on a desk", ["desk", "notebook"])])
    assets = make_resolver(bank, tmp_path, provider=provider).resolve(segments)

    assert assets.by_origin == {"generated": 1}


# --------------------------------------------------------------------------
# stock
# --------------------------------------------------------------------------


def test_cena_generica_vai_para_stock(bank, tmp_path, transcript):
    stock = FakeStock(["city", "office"])
    provider = FakeProvider()
    segments = broll(transcript, [("a city skyline at dusk", ["city", "skyline"])])
    assets = make_resolver(bank, tmp_path, provider=provider, stock=stock).resolve(segments)

    assert assets.by_origin == {"stock": 1}
    assert provider.calls == []
    assert assets.total_cost_usd == 0.0
    assert bank.stats()["by_origin"] == {"stock": 1}


def test_cena_especifica_nao_vai_para_stock(bank, tmp_path, transcript):
    stock = FakeStock(["city", "office"])
    provider = FakeProvider()
    segments = broll(transcript, [("a server rack seen from below", ["server", "rack"])])
    assets = make_resolver(bank, tmp_path, provider=provider, stock=stock).resolve(segments)

    assert assets.by_origin == {"generated": 1}
    assert stock.searches == []


def test_stock_sem_resultado_cai_para_geracao(bank, tmp_path, transcript):
    stock = FakeStock(["city"], hit=False)
    provider = FakeProvider()
    segments = broll(transcript, [("a city skyline at dusk", ["city", "skyline"])])
    assets = make_resolver(bank, tmp_path, provider=provider, stock=stock).resolve(segments)

    assert assets.by_origin == {"generated": 1}
    assert len(provider.calls) == 1


def test_banco_tem_precedencia_sobre_stock(bank, tmp_path, transcript):
    bank.add(path="assets/images/city.png", concept="a city skyline at dusk",
             origin="stock", cost_usd=0.0)
    (tmp_path / "assets" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "assets" / "images" / "city.png").write_bytes(b"x")

    stock = FakeStock(["city"])
    segments = broll(transcript, [("a city skyline at night", ["city", "skyline"])])
    assets = make_resolver(bank, tmp_path, stock=stock).resolve(segments)

    assert assets.by_origin == {"bank": 1}
    assert stock.searches == []


# --------------------------------------------------------------------------
# orcamento
# --------------------------------------------------------------------------


def test_para_antes_de_gastar_se_o_pior_caso_estoura(bank, tmp_path, transcript):
    """Pior caso = tudo gerado. Se estoura o teto, nao gasta um centavo."""
    provider = FakeProvider(cost=0.50)
    segments = broll(transcript, [("a desk with a notebook", ["desk", "notebook"]),
                                  ("an ocean at dawn", ["ocean", "dawn"]),
                                  ("a kitchen counter", ["kitchen", "counter"])])
    with pytest.raises(BudgetExceeded) as excinfo:
        make_resolver(bank, tmp_path, provider=provider, budget=1.00).resolve(segments)

    assert provider.calls == []
    assert excinfo.value.estimate.worst_case_usd == pytest.approx(1.50)
    assert excinfo.value.estimate.within_budget is False


def test_acumulador_interrompe_quando_o_preco_real_e_maior(bank, tmp_path, transcript):
    """O pior caso e o maximo possivel, entao o acumulador so tem serventia
    se o provider cobrar acima do que o config anuncia. Aqui o config diz
    0.05 (pior caso 0.15, dentro do teto de 0.16) e a cobranca real e 0.08:
    depois de duas geracoes o acumulador corta a terceira."""
    provider = FakeProvider(cost=0.05, actual_cost=0.08)
    segments = broll(transcript, [("a desk with a notebook", ["desk", "notebook"]),
                                  ("an ocean at dawn", ["ocean", "dawn"]),
                                  ("a kitchen counter", ["kitchen", "counter"])])
    assets = make_resolver(bank, tmp_path, provider=provider, budget=0.16).resolve(segments)

    assert assets.by_origin == {"generated": 2, "solid": 1}
    assert len(provider.calls) == 2
    assert assets.total_cost_usd == pytest.approx(0.16)
    solid = next(i for i in assets.items if i.origin == "solid")
    assert solid.path is None and "teto" in solid.note


def test_sem_provider_configurado_vira_cor_solida(bank, tmp_path, transcript):
    segments = broll(transcript, [("a desk with a notebook", ["desk", "notebook"])])
    resolver = AssetResolver(
        bank=bank, provider=None, stock=None,
        style_suffix="flat editorial illustration",
        budget_usd=1.50, cost_usd_per_image=0.025,
        images_dir=tmp_path / "assets" / "images",
    )
    assets = resolver.resolve(segments)

    assert assets.by_origin == {"solid": 1}
    assert assets.total_cost_usd == 0.0


def test_estimativa_de_custo_funciona_sem_provider(bank, tmp_path, transcript):
    """O --dry-run precisa estimar custo sem chave de API nenhuma, entao o
    preco unitario sai do config e nao do provider instanciado."""
    segments = broll(transcript, [("a desk with a notebook", ["desk", "notebook"]),
                                  ("an ocean at dawn", ["ocean", "dawn"])])
    resolver = AssetResolver(
        bank=bank, provider=None, stock=None,
        style_suffix="flat editorial illustration",
        budget_usd=1.50, cost_usd_per_image=0.025,
        images_dir=tmp_path / "assets" / "images",
    )
    assets = resolver.resolve(segments, dry_run=True)

    assert assets.estimate.n_to_generate == 2
    assert assets.estimate.worst_case_usd == pytest.approx(0.05)
    assert assets.estimate.estimated_usd == pytest.approx(0.05)


def test_reuso_do_banco_nao_consome_orcamento(bank, tmp_path, transcript):
    for name, concept in [("desk", "a desk with a notebook"), ("ocean", "an ocean at dawn")]:
        bank.add(path=f"assets/images/{name}.png", concept=concept, origin="generated", cost_usd=0.025)
        (tmp_path / "assets" / "images").mkdir(parents=True, exist_ok=True)
        (tmp_path / "assets" / "images" / f"{name}.png").write_bytes(b"x")

    provider = FakeProvider(cost=0.50)
    segments = broll(transcript, [("a notebook on a desk", ["desk", "notebook"]),
                                  ("the ocean at dawn", ["ocean", "dawn"])])
    # pior caso = 2 * 0.50 = 1.00, dentro do teto; nada e gerado de fato
    assets = make_resolver(bank, tmp_path, provider=provider, budget=1.00).resolve(segments)

    assert assets.by_origin == {"bank": 2}
    assert assets.total_cost_usd == 0.0


# --------------------------------------------------------------------------
# dry-run
# --------------------------------------------------------------------------


def test_dry_run_nao_gasta_nem_baixa(bank, tmp_path, transcript):
    stock = FakeStock(["city"])
    provider = FakeProvider()
    segments = broll(transcript, [("a desk with a notebook", ["desk", "notebook"]),
                                  ("a city skyline at dusk", ["city", "skyline"])])
    assets = make_resolver(bank, tmp_path, provider=provider, stock=stock).resolve(
        segments, dry_run=True)

    assert assets.dry_run is True
    assert provider.calls == [] and stock.searches == []
    assert assets.total_cost_usd == 0.0
    assert bank.stats()["assets"] == 0
    # e mostra o prompt que seria usado
    generated = next(i for i in assets.items if i.origin == "generated")
    assert "flat editorial illustration" in generated.prompt


def test_dry_run_estima_a_divisao_entre_rotas(bank, tmp_path, transcript):
    bank.add(path="assets/images/desk.png", concept="a desk with a notebook",
             origin="generated", cost_usd=0.025)
    (tmp_path / "assets" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "assets" / "images" / "desk.png").write_bytes(b"x")

    stock = FakeStock(["city"])
    segments = broll(transcript, [("a notebook on a desk", ["desk", "notebook"]),
                                  ("a city skyline at dusk", ["city", "skyline"]),
                                  ("a server rack seen from below", ["server", "rack"])])
    assets = make_resolver(bank, tmp_path, stock=stock).resolve(segments, dry_run=True)

    assert assets.estimate.n_broll == 3
    assert assets.estimate.n_from_bank == 1
    assert assets.estimate.n_from_stock == 1
    assert assets.estimate.n_to_generate == 1
    assert assets.estimate.estimated_usd == pytest.approx(0.025)
    assert assets.estimate.worst_case_usd == pytest.approx(0.075)


# --------------------------------------------------------------------------
# manutencao
# --------------------------------------------------------------------------


def test_stats_do_banco(bank, tmp_path):
    a = bank.add(path="assets/images/a.png", concept="a desk", origin="generated", cost_usd=0.025)
    bank.add(path="assets/images/b.png", concept="a city", origin="stock", cost_usd=0.0)
    bank.mark_used(a.id)
    bank.mark_used(a.id)

    stats = bank.stats()
    assert stats["assets"] == 2
    assert stats["by_origin"] == {"generated": 1, "stock": 1}
    assert stats["total_spent_usd"] == pytest.approx(0.025)
    assert stats["total_uses"] == 2
    assert stats["never_used"] == 1


def test_prune_remove_linha_e_arquivo(bank, tmp_path):
    images = tmp_path / "assets" / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / "old.png").write_bytes(b"x")
    bank.add(path="assets/images/old.png", concept="a desk", origin="generated", cost_usd=0.025)
    # envelhece a linha na mao
    bank.conn.execute("UPDATE assets SET created_at = datetime('now', '-200 days')")
    bank.conn.commit()

    removed = bank.prune(unused_days=90)
    assert removed == ["assets/images/old.png"]
    assert not (images / "old.png").exists()
    assert bank.stats()["assets"] == 0


def test_prune_preserva_asset_recente(bank, tmp_path):
    images = tmp_path / "assets" / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / "new.png").write_bytes(b"x")
    bank.add(path="assets/images/new.png", concept="a desk", origin="generated", cost_usd=0.025)

    assert bank.prune(unused_days=90) == []
    assert bank.stats()["assets"] == 1


def test_banco_vazio_nao_carrega_o_embedder(bank, tmp_path, transcript):
    """O embedder arrasta o torch. Com banco vazio nao ha o que comparar, e o
    --dry-run do primeiro video nao pode esperar o modelo carregar para nada."""

    class ExplodingEmbedder:
        def embed(self, text):
            raise AssertionError("o embedder foi carregado com o banco vazio")

    bank.embedder = ExplodingEmbedder()
    assert bank.find_similar("a desk with a notebook") is None


def test_cosine_em_vetores_de_tamanho_diferente_nao_explode():
    assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
