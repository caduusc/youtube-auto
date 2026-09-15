"""Estagio 4: resolve uma imagem para cada segmento de b-roll."""

from __future__ import annotations

from ..bank import AssetBank
from ..config import Config
from ..embed import SentenceTransformerEmbedder
from ..log import log, stage
from ..providers import PexelsStock, build_provider
from ..resolver import AssetResolver, BudgetExceeded
from ..schemas import EDL, Assets, Manifest
from ..util import read_json_if_fresh, text_hash, write_json


def open_bank(config: Config) -> AssetBank:
    return AssetBank(
        root=config.root,
        db_path=config.path(config.bank.db_path),
        images_dir=config.path(config.bank.images_dir),
        embedder=SentenceTransformerEmbedder(config.bank.embedding_model),
        threshold=config.bank.similarity_threshold,
    )


def build_resolver(config: Config, bank: AssetBank, *, dry_run: bool) -> AssetResolver:
    """Em dry-run nada e chamado, entao a falta de chave nao pode impedir o
    modo que existe justamente para calibrar antes de gastar."""
    provider = stock = None

    try:
        provider = build_provider(config.image_provider.active, config.image_provider)
    except RuntimeError as exc:
        if not dry_run:
            raise
        log("assets.warn", detail=f"provider indisponivel ({exc}); dry-run segue")

    try:
        stock = PexelsStock(config.stock)
    except RuntimeError as exc:
        if not dry_run:
            log("assets.warn", detail=f"stock indisponivel ({exc}); tudo vai para geracao")
        else:
            log("assets.warn", detail=f"stock indisponivel ({exc}); dry-run segue")

    active = config.image_provider.replicate  # o unico provider implementado
    return AssetResolver(
        bank=bank,
        provider=provider,
        stock=stock,
        style_suffix=config.style_suffix,
        budget_usd=config.budget.max_usd_per_video,
        cost_usd_per_image=active.cost_usd_per_image,
        images_dir=config.path(config.bank.images_dir),
        image_extension=active.output_format,
    )


def cache_key(edl: EDL, config: Config) -> str:
    """Identidade dos assets: a EDL MAIS tudo que decide qual imagem sai dela.

    `style_suffix` esta aqui porque o README manda calibra-lo — e sem ele na
    chave, mexer no campo mais importante do config nao regera nada.
    `similarity_threshold` e `generic_tags` mudam a rota de cada segmento
    (banco / stock / geracao), e o modelo do provider muda a imagem.
    """
    return text_hash(
        edl.digest(),
        config.style_suffix,
        config.bank.similarity_threshold,
        ",".join(sorted(config.stock.generic_tags)),
        config.image_provider.active,
        config.image_provider.replicate.model,
    )


def run(manifest: Manifest, edl: EDL, config: Config, *, dry_run: bool = False) -> Assets:
    work = config.work_dir / manifest.slug
    assets_path = work / ("assets.dryrun.json" if dry_run else "assets.json")
    key = cache_key(edl, config)

    cached = read_json_if_fresh(assets_path, Assets, key)
    if cached is not None and cached.dry_run == dry_run:
        log("assets.cached", slug=manifest.slug, cost=f"${cached.total_cost_usd:.2f}")
        return cached

    with stage("assets", n_broll=edl.stats.n_broll, dry_run=dry_run):
        bank = open_bank(config)
        try:
            resolver = build_resolver(config, bank, dry_run=dry_run)
            try:
                assets = resolver.resolve(edl.segments, dry_run=dry_run)
            except BudgetExceeded as exc:
                print_estimate(exc.estimate, config)
                raise
            assets.input_hash = key
            write_json(assets_path, assets)
            log("assets.ok", cost=f"${assets.total_cost_usd:.4f}", **assets.by_origin)
            return assets
        finally:
            bank.close()


def print_estimate(estimate, config: Config) -> None:
    """Relatorio de estimativa. E o que o dry-run existe para mostrar."""
    rate = config.budget.brl_per_usd
    print()
    print("  Estimativa de custo")
    print("  " + "-" * 52)
    print(f"  segmentos de b-roll        {estimate.n_broll:>6}")
    print(f"    do banco (custo zero)    {estimate.n_from_bank:>6}")
    print(f"    do stock (custo zero)    {estimate.n_from_stock:>6}")
    print(f"    a gerar                  {estimate.n_to_generate:>6}")
    print("  " + "-" * 52)
    print(f"  custo estimado             USD {estimate.estimated_usd:>8.4f}   "
          f"BRL {estimate.estimated_usd * rate:>8.2f}")
    print(f"  pior caso (tudo gerado)    USD {estimate.worst_case_usd:>8.4f}   "
          f"BRL {estimate.worst_case_usd * rate:>8.2f}")
    print(f"  teto por video             USD {estimate.budget_usd:>8.4f}   "
          f"BRL {estimate.budget_usd * rate:>8.2f}")
    print("  " + "-" * 52)
    if estimate.within_budget:
        reuse = (estimate.n_from_bank + estimate.n_from_stock) / estimate.n_broll if estimate.n_broll else 0.0
        print(f"  cabe no orcamento. reuso sem custo: {reuse:.0%}")
    else:
        print("  NAO CABE: o pior caso estoura o teto, nada foi gasto.")
        print("  Ajuste budget.max_usd_per_video, ou troque o modelo do provider")
        print("  por um mais barato, ou amplie stock.generic_tags.")
    print()
