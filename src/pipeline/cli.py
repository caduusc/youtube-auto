"""Interface de linha de comando.

    pipeline run <arquivo.mp4> [--config config.yaml] [--dry-run] [--from <estagio>]
    pipeline <estagio> <slug>          # roda um estagio isolado
    pipeline bank stats
    pipeline bank prune --unused-days 90
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import stages
from .config import Config, load_config
from .log import log
from .resolver import BudgetExceeded
from .schemas import EDL, Assets, Manifest, Transcript
from .stages.assets import open_bank
from .util import read_json


# --------------------------------------------------------------------------
# artefatos em disco
# --------------------------------------------------------------------------


def work_dir(slug: str, config: Config) -> Path:
    path = config.work_dir / slug
    if not path.exists():
        available = sorted(p.name for p in config.work_dir.glob("*") if p.is_dir())
        listing = "\n  ".join(available) if available else "(nenhum)"
        raise SystemExit(f"slug '{slug}' nao existe em {config.work_dir}\n\nDisponiveis:\n  {listing}")
    return path


def load(slug: str, config: Config, what: str):
    """Le um artefato, com mensagem util se o estagio anterior nao rodou."""
    work = work_dir(slug, config)
    table = {
        "manifest": ("manifest.json", Manifest, "ingest"),
        "transcript": ("transcript.json", Transcript, "transcribe"),
        "edl": ("edl.json", EDL, "plan"),
        "assets": ("assets.json", Assets, "assets"),
    }
    filename, model, producer = table[what]
    path = work / filename
    if not path.exists():
        raise SystemExit(f"{filename} nao existe em {work} — rode antes: pipeline {producer} {slug}")
    return read_json(path, model)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def cmd_run(args, config: Config) -> int:
    start = args.start or "ingest"
    if start not in stages.ORDER:
        raise SystemExit(f"estagio desconhecido: {start}\nOpcoes: {', '.join(stages.ORDER)}")
    from_index = stages.ORDER.index(start)

    # --dry-run vai ate o estagio 4 e nao gasta nada
    last_index = stages.ORDER.index("assets") if args.dry_run else len(stages.ORDER) - 1
    if from_index > last_index:
        raise SystemExit(f"--dry-run nao chega ao estagio '{start}'")

    source = Path(args.input).resolve()

    if from_index == 0:
        manifest = stages.ingest.run(source, config)
    else:
        # retomada: o slug sai do proprio arquivo de entrada
        from .stages.ingest import slug_for
        from .util import file_hash

        manifest = load(slug_for(source, file_hash(source)), config, "manifest")
        log("run.resume", slug=manifest.slug, at=start)

    slug = manifest.slug

    transcript = stages.transcribe.run(manifest, config) if from_index <= 1 else load(slug, config, "transcript")
    edl = stages.plan.run(manifest, transcript, config) if from_index <= 2 else load(slug, config, "edl")

    try:
        assets = (
            stages.assets.run(manifest, edl, config, dry_run=args.dry_run)
            if from_index <= 3
            else load(slug, config, "assets")
        )
    except BudgetExceeded:
        return 2

    if args.dry_run:
        print_dry_run(edl, assets, config)
        return 0

    output = stages.render.run(manifest, transcript, edl, assets, config)
    stages.report.run(manifest, edl, assets, output, config)
    return 0


def print_dry_run(edl: EDL, assets: Assets, config: Config) -> None:
    """O relatorio de calibragem: EDL, prompts e a divisao de custo."""
    from .stages.assets import print_estimate

    print()
    print("  EDL")
    print("  " + "-" * 72)
    by_index = {item.segment_index: item for item in assets.items}
    for index, segment in enumerate(edl.segments):
        window = f"{segment.start:7.1f}s -{segment.end:7.1f}s"
        if segment.kind == "aroll":
            print(f"  {window}  aroll  ({segment.duration:4.1f}s)")
            continue
        item = by_index.get(index)
        origin = item.origin if item else "?"
        tail = ""
        if item and item.origin == "bank":
            tail = f"  <- banco #{item.asset_id} (cos {item.similarity:.3f})"
        elif item and item.origin == "stock":
            tail = f"  <- stock: {' '.join(segment.concept_tags)}"
        print(f"  {window}  broll  ({segment.duration:4.1f}s)  [{origin}]{tail}")
        print(f"                       concept: {segment.concept}")
        print(f"                       tags:    {', '.join(segment.concept_tags)}")
        if item and item.origin == "generated" and item.prompt:
            print(f"                       prompt:  {item.prompt}")
    print("  " + "-" * 72)
    print(f"  b-roll cobre {edl.stats.broll_ratio:.0%} do video, "
          f"{edl.stats.switches_per_minute:.1f} trocas/min "
          f"(pico de {edl.stats.switches_per_minute_max:.0f} numa janela de 60s)")
    print_estimate(assets.estimate, config)
    print("  Nada foi baixado, gerado, nem cobrado.")
    print()


# --------------------------------------------------------------------------
# estagios isolados
# --------------------------------------------------------------------------


def cmd_stage(name: str, slug: str, config: Config) -> int:
    if name == "ingest":
        manifest = load(slug, config, "manifest")
        stages.ingest.run(Path(manifest.source_path), config)
    elif name == "transcribe":
        stages.transcribe.run(load(slug, config, "manifest"), config)
    elif name == "plan":
        stages.plan.run(load(slug, config, "manifest"), load(slug, config, "transcript"), config)
    elif name == "assets":
        try:
            stages.assets.run(load(slug, config, "manifest"), load(slug, config, "edl"), config)
        except BudgetExceeded:
            return 2
    elif name == "render":
        stages.render.run(
            load(slug, config, "manifest"), load(slug, config, "transcript"),
            load(slug, config, "edl"), load(slug, config, "assets"), config,
        )
    elif name == "report":
        manifest = load(slug, config, "manifest")
        stages.report.run(
            manifest, load(slug, config, "edl"), load(slug, config, "assets"),
            config.work_dir / slug / "final.mp4", config,
        )
    return 0


# --------------------------------------------------------------------------
# bank
# --------------------------------------------------------------------------


def cmd_bank(args, config: Config) -> int:
    bank = open_bank(config)
    try:
        if args.bank_command == "stats":
            stats = bank.stats()
            print()
            print("  Banco de assets")
            print("  " + "-" * 52)
            print(f"  imagens                    {stats['assets']:>6}")
            for origin, count in sorted(stats["by_origin"].items()):
                print(f"    {origin:<24} {count:>6}")
            print(f"  nunca usadas               {stats['never_used']:>6}")
            outros = {m: n for m, n in stats["by_model"].items() if m != stats["active_model"]}
            if outros:
                print("  " + "-" * 52)
                print(f"  modelo ativo: {stats['active_model']}")
                print("  imagens de OUTRO modelo (nao serao reusadas):")
                for model, count in sorted(outros.items()):
                    print(f"    {model:<34} {count:>6}")
            print(f"  usos totais                {stats['total_uses']:>6}")
            print(f"  usos por imagem            {stats['avg_uses_per_asset']:>6.2f}")
            print("  " + "-" * 52)
            spent = stats["total_spent_usd"]
            print(f"  gasto acumulado            USD {spent:>8.4f}   "
                  f"BRL {spent * config.budget.brl_per_usd:>8.2f}")
            if stats["total_uses"]:
                print(f"  custo por uso              USD {spent / stats['total_uses']:>8.4f}")
            print(f"  {stats['db_path']}")
            print()
        elif args.bank_command == "prune":
            removed = bank.prune(args.unused_days)
            print(f"removidas {len(removed)} imagens sem uso ha {args.unused_days} dias")
            for path in removed:
                print(f"  {path}")
    finally:
        bank.close()
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml", help="default: config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="roda o pipeline inteiro")
    run_cmd.add_argument("input", help="arquivo de video de entrada")
    run_cmd.add_argument("--dry-run", action="store_true",
                         help="vai ate o estagio 4 sem gastar: mostra EDL, prompts e custo estimado")
    run_cmd.add_argument("--from", dest="start", metavar="ESTAGIO",
                         help=f"comeca deste estagio ({', '.join(stages.ORDER)})")

    for name in stages.ORDER:
        stage_cmd = sub.add_parser(name, help=f"roda o estagio {name} isolado")
        stage_cmd.add_argument("slug")

    bank_cmd = sub.add_parser("bank", help="banco de assets")
    bank_sub = bank_cmd.add_subparsers(dest="bank_command", required=True)
    bank_sub.add_parser("stats", help="estatisticas do banco")
    prune_cmd = bank_sub.add_parser("prune", help="remove imagens sem uso")
    prune_cmd.add_argument("--unused-days", type=int, default=90)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(
            f"config nao encontrado: {config_path}\n"
            "Copie o exemplo:  cp config.example.yaml config.yaml"
        )
    config = load_config(config_path)

    try:
        if args.command == "run":
            return cmd_run(args, config)
        if args.command == "bank":
            return cmd_bank(args, config)
        return cmd_stage(args.command, args.slug, config)
    except KeyboardInterrupt:
        log("interrupted", detail="os artefatos ja escritos continuam validos; rode de novo para retomar")
        return 130


if __name__ == "__main__":
    sys.exit(main())
