"""Resolucao de imagem para cada segmento de b-roll.

Precedencia, parando na primeira que atender:

  1. banco local   — custo zero, embedding + cosseno acima do limiar
  2. stock livre   — custo zero, so para cena generica (tags do config)
  3. geracao       — custo real, `concept` + `style_suffix`

O orcamento entra em dois pontos:

  * um teto de pior caso antes de comecar — se "tudo gerado" estouraria, nao
    gasta um centavo e devolve a estimativa;
  * um acumulador durante a execucao, que soma o custo REAL devolvido pelo
    provider. Como o pior caso ja e o maximo possivel, o acumulador so
    dispara quando o preco real vem acima do `cost_usd_per_image` do config
    — provider que reajustou, cobranca por passo, retry cobrado duas vezes.
    E exatamente por isso que ele soma o retorno de `generate()` e nao o
    valor estimado. O que sobra depois do teto vira cor solida.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .bank import AssetBank
from .log import log
from .providers import ImageProvider, PexelsStock
from .schemas import AssetEstimate, AssetItem, Assets, EDLSegment, StoryboardBeat
from .util import text_hash


class BudgetExceeded(RuntimeError):
    """Pior caso acima do teto. Carrega a estimativa para o relatorio."""

    def __init__(self, estimate: AssetEstimate) -> None:
        super().__init__(
            f"pior caso de USD {estimate.worst_case_usd:.4f} acima do teto de "
            f"USD {estimate.budget_usd:.4f}"
        )
        self.estimate = estimate


@dataclass(frozen=True)
class ImageRequest:
    """Um pedido de imagem: uma identidade, um conceito e as tags.

    Nem EDL nem storyboard — os dois sabem virar isto. O resolver nunca
    precisou de uma EDL: ele usava tres campos dela (`kind`, `concept`,
    `concept_tags`) e carregava o indice adiante apenas como identidade.
    Explicitar isso e o que faz o MESMO resolver servir aos dois caminhos, em
    vez de existirem duas resolucoes que divergem no reuso, no orcamento e no
    fallback de stock — e divergiriam em silencio, porque cada uma passaria
    nos proprios testes.

    `keyed_by` diz o que `key` significa, e com isso em qual campo do
    `AssetItem` ela vai. Ver o docstring de `AssetItem`.
    """

    key: int
    concept: str
    concept_tags: list[str]
    keyed_by: Literal["segment", "beat"] = "segment"


def from_edl(segments: list[EDLSegment]) -> list[ImageRequest]:
    """Os pedidos de uma EDL: um por faixa de b-roll, na chave do indice.

    O indice e a POSICAO na lista de segmentos, e nao `first_segment`: e por
    ela que o render junta imagem e faixa.
    """
    return [
        ImageRequest(key=index, concept=s.concept, concept_tags=list(s.concept_tags))
        for index, s in enumerate(segments) if s.kind == "broll"
    ]


def from_storyboard(beats: list[StoryboardBeat]) -> list[ImageRequest]:
    """Os pedidos de um storyboard: um por beat, na chave do `beat_id`.

    Um pedido por beat e nao por sub-plano: os sub-planos saem todos da mesma
    imagem, recortando regioes diferentes dela. E o que faz uma faixa de 10s
    custar uma imagem em vez de quatro.
    """
    return [
        ImageRequest(key=b.beat_id, concept=b.concept,
                     concept_tags=list(b.concept_tags), keyed_by="beat")
        for b in beats
    ]


@dataclass
class Decision:
    """O que fazer com um pedido, decidido sem gastar nada."""

    request: ImageRequest
    route: str                      # "bank" | "stock" | "generate"
    asset_id: int | None = None
    path: str | None = None
    similarity: float | None = None

    @property
    def key(self) -> int:
        return self.request.key

    @property
    def identity(self) -> dict[str, int]:
        """O campo de identidade do `AssetItem` deste pedido."""
        if self.request.keyed_by == "beat":
            return {"beat_id": self.request.key}
        return {"segment_index": self.request.key}


class AssetResolver:
    def __init__(
        self,
        *,
        bank: AssetBank,
        provider: ImageProvider | None,
        stock: PexelsStock | None,
        style_suffix: str,
        budget_usd: float,
        cost_usd_per_image: float,
        images_dir: Path,
        image_extension: str = "png",
    ) -> None:
        self.bank = bank
        self.provider = provider
        self.stock = stock
        self.style_suffix = style_suffix
        self.budget_usd = budget_usd
        # O preco unitario vem do config, nao do provider: o --dry-run existe
        # para estimar custo e precisa funcionar sem chave de API nenhuma.
        self.cost_usd_per_image = cost_usd_per_image
        self.images_dir = Path(images_dir)
        self.image_extension = image_extension

    # -- prompt -----------------------------------------------------------

    def prompt_for(self, request: ImageRequest) -> str:
        return f"{request.concept.strip()}, {self.style_suffix.strip()}"

    # -- fase 1: decidir (nao gasta) --------------------------------------

    def plan(self, requests: list[ImageRequest]) -> list[Decision]:
        """Classifica cada pedido. Consultas ao banco acontecem de verdade
        aqui — sao gratis, e sem elas a estimativa nao vale nada."""
        decisions: list[Decision] = []
        used_ids: set[int] = set()

        for request in requests:
            hit = self.bank.find_similar(request.concept, exclude_ids=used_ids)
            if hit is not None:
                asset, score = hit
                used_ids.add(asset.id)
                decisions.append(
                    Decision(request, "bank", asset_id=asset.id,
                             path=asset.path, similarity=score)
                )
                continue

            if self.stock is not None and self.stock.is_generic(request.concept_tags):
                decisions.append(Decision(request, "stock"))
                continue

            decisions.append(Decision(request, "generate"))

        return decisions

    def estimate(self, decisions: list[Decision]) -> AssetEstimate:
        unit = self.cost_usd_per_image
        n_broll = len(decisions)
        n_generate = sum(1 for d in decisions if d.route == "generate")
        worst_case = n_broll * unit
        return AssetEstimate(
            n_broll=n_broll,
            n_from_bank=sum(1 for d in decisions if d.route == "bank"),
            n_from_stock=sum(1 for d in decisions if d.route == "stock"),
            n_to_generate=n_generate,
            worst_case_usd=round(worst_case, 4),
            estimated_usd=round(n_generate * unit, 4),
            budget_usd=self.budget_usd,
            within_budget=worst_case <= self.budget_usd,
        )

    # -- fase 2: executar --------------------------------------------------

    def resolve(self, requests: list[ImageRequest], *, dry_run: bool = False) -> Assets:
        decisions = self.plan(requests)
        estimate = self.estimate(decisions)

        log("assets.estimate", n_broll=estimate.n_broll, bank=estimate.n_from_bank,
            stock=estimate.n_from_stock, generate=estimate.n_to_generate,
            worst_case=f"${estimate.worst_case_usd:.2f}",
            estimated=f"${estimate.estimated_usd:.2f}", budget=f"${estimate.budget_usd:.2f}")

        if not estimate.within_budget:
            raise BudgetExceeded(estimate)

        if dry_run:
            return self._dry_run_assets(decisions, estimate)

        items: list[AssetItem] = []
        spent = 0.0
        unit = self.cost_usd_per_image

        for decision in decisions:
            if decision.route == "bank":
                self.bank.mark_used(decision.asset_id)
                items.append(AssetItem(
                    **decision.identity, origin="bank",
                    path=decision.path, asset_id=decision.asset_id,
                    similarity=round(decision.similarity, 4), cost_usd=0.0,
                ))
                continue

            if decision.route == "stock":
                item = self._try_stock(decision)
                if item is not None:
                    items.append(item)
                    continue
                # stock nao tinha nada utilizavel; cai para geracao
                log("assets.stock_fallthrough", pedido=decision.key)

            # geracao — checa o acumulador antes de gastar
            if self.provider is None:
                items.append(self._solid(decision, "sem provider de imagem configurado"))
                continue
            if spent + unit > self.budget_usd:
                items.append(self._solid(
                    decision,
                    f"teto de USD {self.budget_usd:.2f} atingido (gasto: "
                    f"USD {spent:.2f})"))
                continue
            item = self._generate(decision)
            items.append(item)
            spent += item.cost_usd  # custo real, nao o estimado

        items.sort(key=lambda i: max(i.segment_index, i.beat_id))
        by_origin: dict[str, int] = {}
        for item in items:
            by_origin[item.origin] = by_origin.get(item.origin, 0) + 1

        return Assets(
            input_hash="",  # preenchido pelo estagio
            dry_run=False,
            items=items,
            estimate=estimate,
            total_cost_usd=round(sum(i.cost_usd for i in items), 4),
            by_origin=by_origin,
        )

    # -- helpers -----------------------------------------------------------

    def _dry_run_assets(self, decisions: list[Decision], estimate: AssetEstimate) -> Assets:
        items = [
            AssetItem(
                **d.identity,
                origin="bank" if d.route == "bank" else ("stock" if d.route == "stock" else "generated"),
                path=d.path,
                asset_id=d.asset_id,
                similarity=round(d.similarity, 4) if d.similarity is not None else None,
                prompt=self.prompt_for(d.request) if d.route == "generate" else None,
                cost_usd=0.0,
                note="dry-run: nada foi baixado nem gerado",
            )
            for d in decisions
        ]
        by_origin: dict[str, int] = {}
        for item in items:
            by_origin[item.origin] = by_origin.get(item.origin, 0) + 1
        return Assets(input_hash="", dry_run=True, items=items, estimate=estimate,
                      total_cost_usd=0.0, by_origin=by_origin)

    def _destination(self, prompt: str) -> Path:
        from .util import now_iso

        name = f"{text_hash(prompt, now_iso())[:16]}.{self.image_extension}"
        return self.images_dir / name

    def _index(
        self, decision: Decision, destination: Path, *, origin: str,
        prompt: str | None, cost_usd: float,
    ) -> tuple[int | None, str | None]:
        """Registra o asset no banco. Devolve (id, aviso).

        A imagem ja esta em disco e, se foi gerada, ja foi paga. Se indexar
        falhar, o asset NAO pode ser descartado: o video desta rodada ainda
        usa o arquivo, e a linha fica no banco com embedding vazio para o
        custo continuar contabilizado. Embedding vazio nunca casa em
        `find_similar` (o cosseno devolve 0 para tamanhos diferentes), entao
        a linha nao polui o reuso.
        """
        try:
            embedding = self.bank.embedder.embed(decision.request.concept)
            warning = None
        except Exception as exc:
            embedding = []
            warning = f"asset salvo sem embedding ({type(exc).__name__}); nao sera reusado"
            log("assets.index_degraded", pedido=decision.key, error=repr(exc))

        asset = self.bank.add(
            path=self.bank.relative(destination), concept=decision.request.concept,
            origin=origin, prompt=prompt, cost_usd=cost_usd, embedding=embedding,
        )
        self.bank.mark_used(asset.id)
        return asset.id, warning

    def _solid(self, decision: Decision, reason: str) -> AssetItem:
        log("assets.solid", pedido=decision.key, reason=reason)
        return AssetItem(**decision.identity, origin="solid",
                         path=None, cost_usd=0.0, note=reason)

    def _try_stock(self, decision: Decision) -> AssetItem | None:
        if self.stock is None:
            return None
        log("assets.stock_search", pedido=decision.key,
            tags=" ".join(decision.request.concept_tags))
        try:
            found = self.stock.search(decision.request.concept_tags)
        except Exception as exc:
            log("assets.stock_error", pedido=decision.key, error=repr(exc))
            return None
        if found is None:
            return None
        url, credit = found
        destination = self._destination(" ".join(decision.request.concept_tags))
        log("assets.stock_download", pedido=decision.key, file=destination.name)
        self.stock.download(url, destination)
        asset_id, warning = self._index(
            decision, destination, origin="stock", prompt=credit, cost_usd=0.0)
        log("assets.stock", pedido=decision.key, file=destination.name)
        return AssetItem(**decision.identity, origin="stock",
                         path=self.bank.relative(destination), asset_id=asset_id,
                         prompt=credit, cost_usd=0.0, note=warning)

    def _generate(self, decision: Decision) -> AssetItem:
        prompt = self.prompt_for(decision.request)
        destination = self._destination(prompt)
        cost = self.provider.generate(prompt, destination)
        asset_id, warning = self._index(
            decision, destination, origin="generated", prompt=prompt, cost_usd=cost)
        return AssetItem(**decision.identity, origin="generated",
                         path=self.bank.relative(destination), asset_id=asset_id,
                         prompt=prompt, cost_usd=round(cost, 4), note=warning)
