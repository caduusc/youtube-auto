"""Interface de linha de comando.

    pipeline run <arquivo.mp4> [--config config.yaml] [--dry-run] [--from <estagio>]
    pipeline <estagio> <slug>          # roda um estagio isolado
    pipeline script "a sua ideia"      # roteiro primeiro
    pipeline storyboard <slug>
    pipeline images <slug> [--dry-run]
    pipeline approve script|storyboard|images <slug>
    pipeline shoot <slug> <arquivo.mp4>    # depois de gravar, roda sozinho
    pipeline align <slug>
    pipeline status
    pipeline ui                        # a mesma revisao, no navegador
    pipeline bank stats
    pipeline bank prune --unused-days 90
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import approval, progress, stages
from .config import Config, load_config
from .log import log
from .resolver import BudgetExceeded
from .schemas import EDL, Assets, Manifest, Placements, Storyboard, Transcript, TrimPlan
from .script import read as read_script, spoken_seconds, words
from .stages.assets import open_bank
from .storyboard import render_text as render_storyboard
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
        # O estagio 3 em diante trabalha na timeline cortada.
        "trimmed": ("transcript.trimmed.json", Transcript, "trim"),
        "trim": ("trim.json", TrimPlan, "trim"),
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

    transcript = (
        stages.transcribe.run(manifest, config) if from_index <= 1
        else load(slug, config, "transcript")
    )

    if from_index <= 2:
        trim_plan, trimmed = stages.trim.run(manifest, transcript, config)
    else:
        trim_plan, trimmed = load(slug, config, "trim"), load(slug, config, "trimmed")

    edl = (
        stages.plan.run(manifest, trimmed, config) if from_index <= 3
        else load(slug, config, "edl")
    )

    try:
        assets = (
            stages.assets.run(manifest, edl, config, dry_run=args.dry_run)
            if from_index <= 4
            else load(slug, config, "assets")
        )
    except BudgetExceeded:
        return 2

    if args.dry_run:
        print(stages.trim.render_text(trim_plan))
        print_dry_run(edl, assets, config)
        return 0

    output = stages.render.run(manifest, trimmed, edl, assets, config, trim_plan)
    stages.report.run(manifest, edl, assets, output, config)
    return 0


def print_dry_run(edl: EDL, assets: Assets, config: Config) -> None:
    """O relatorio de calibragem: EDL, prompts e a divisao de custo."""
    from .stages.assets import print_estimate

    print()
    if edl.brief is not None:
        b = edl.brief
        print("  Briefing visual")
        print("  " + "-" * 72)
        print(f"  assunto:    {b.subject}")
        print(f"  argumento:  {b.argument}")
        print(f"  espectador: {b.audience_takeaway}")
        print(f"  vocabulario: {', '.join(b.visual_vocabulary)}")
        print(f"  evitar:      {', '.join(b.avoid)}")
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
    elif name == "trim":
        _, plan = stages.trim.run(
            load(slug, config, "manifest"), load(slug, config, "transcript"), config)
        print(stages.trim.render_text(load(slug, config, "trim")))
    elif name == "plan":
        stages.plan.run(load(slug, config, "manifest"), load(slug, config, "trimmed"), config)
    elif name == "assets":
        try:
            stages.assets.run(load(slug, config, "manifest"), load(slug, config, "edl"), config)
        except BudgetExceeded:
            return 2
    elif name == "render":
        stages.render.run(
            load(slug, config, "manifest"), load(slug, config, "trimmed"),
            load(slug, config, "edl"), load(slug, config, "assets"), config,
            load(slug, config, "trim"),
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
            # Mesmo aviso do modelo, pela mesma razao: imagem que nao pode
            # ser reusada continua ocupando disco, e sem isso o `stats` diria
            # que o banco tem 40 imagens quando so 6 servem.
            estilos = {s: n for s, n in stats["by_style"].items() if s != stats["active_style"]}
            if estilos:
                print("  " + "-" * 52)
                print(f"  estilo ativo: {stats['active_style'] or '(nenhum)'}")
                print("  geradas em OUTRO estilo (nao serao reusadas):")
                for style, count in sorted(estilos.items()):
                    rotulo = "? (antes da coluna de estilo)" if style == "?" else style
                    print(f"    {rotulo:<34} {count:>6}")
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
# roteiro primeiro
# --------------------------------------------------------------------------


def cmd_script(args, config: Config) -> int:
    """A ideia vira roteiro. Sem gravacao, sem midia."""
    slug = args.slug or stages.script.slug_for(args.idea)
    script = stages.script.run(
        args.idea, slug=slug, target_seconds=args.seconds, config=config,
    )
    caminho = config.work_dir / slug / "script.md"
    falado = spoken_seconds(script, config.script.words_per_minute)

    print()
    print(f"  {caminho}")
    print(f"  {len(script.beats)} beats, {words(script)} palavras, "
          f"~{falado:.0f}s falados (alvo {args.seconds:.0f}s)")
    print()
    print("  Revise o arquivo — editar e esperado — e aprove com:")
    print(f"    pipeline approve script {slug}")
    print()
    return 0


def cmd_storyboard(args, config: Config) -> int:
    script = read_script(config.work_dir / args.slug / "script.md")
    board = stages.storyboard.run(script, config=config)
    print(render_storyboard(board, script))
    print("  Revise os conceitos e aprove com:")
    print(f"    pipeline approve storyboard {args.slug}")
    print()
    return 0


def cmd_images(args, config: Config) -> int:
    """Estagio 3: o storyboard aprovado vira imagens. Gasta dinheiro."""
    work = config.work_dir / args.slug
    script = read_script(work / "script.md")
    board = load_storyboard(work, args.slug)

    try:
        assets = stages.images.run(script, board, config, dry_run=args.dry_run)
    except BudgetExceeded:
        return 2

    print(stages.images.render_text(assets, board, config))
    if args.dry_run:
        print("  Nada foi baixado, gerado, nem cobrado.\n")
        return 0
    print("  Revise as imagens e aprove com:")
    print(f"    pipeline approve images {args.slug}\n")
    return 0


def cmd_regenerate(args, config: Config) -> int:
    """Troca a imagem de um beat por uma nova. Gasta.

    Existe na CLI e nao so na UI de proposito: a promessa e que a CLI faz tudo
    sozinha, e um comando que so existe no navegador quebra isso.
    """
    work = work_dir(args.slug, config)
    script = read_script(work / "script.md")
    board = load_storyboard(work, args.slug)

    try:
        assets = stages.images.regenerate(
            script, board, read_json(work / "assets.json", Assets),
            args.beat, config)
    except BudgetExceeded:
        return 2

    print(stages.images.render_text(assets, board, config))
    print("  A aprovacao das imagens foi revogada. Revise e aprove de novo:")
    print(f"    pipeline approve images {args.slug}\n")
    return 0


def cmd_align(args, config: Config) -> int:
    """Estagio 6: onde cada imagem cai na fala gravada."""
    work = config.work_dir / args.slug
    script = read_script(work / "script.md")
    board = load_storyboard(work, args.slug)
    edl, _ = stages.align.run(
        script, board, load(args.slug, config, "trimmed"),
        read_json(work / "assets.json", Assets), config,
    )
    print(stages.align.render_text(read_json(work / "align.json", Placements), board))
    print(f"  {edl.stats.n_broll} faixas de b-roll, "
          f"{edl.stats.broll_ratio:.0%} de cobertura\n")
    return 0


def cmd_shoot(args, config: Config) -> int:
    """Depois de gravar: do arquivo ao video montado, sem parar.

    O slug vem antes do arquivo de proposito. E ele que liga a gravacao ao
    roteiro, ao storyboard e as imagens que voce ja aprovou — sem ele o
    `ingest` abriria um `work/` novo derivado do nome do arquivo, e o `align`
    nao acharia nada. Ver `stages.ingest.run`.
    """
    work = work_dir(args.slug, config)
    script = read_script(work / "script.md")
    board = load_storyboard(work, args.slug)
    assets = read_json(work / "assets.json", Assets)

    manifest = stages.ingest.run(Path(args.input), config, slug=args.slug)
    transcript = stages.transcribe.run(manifest, config)
    trim_plan, trimmed = stages.trim.run(manifest, transcript, config)
    print(stages.trim.render_text(trim_plan))

    edl, aligned = stages.align.run(script, board, trimmed, assets, config)
    print(stages.align.render_text(read_json(work / "align.json", Placements), board))

    output = stages.render.run(manifest, trimmed, edl, aligned, config, trim_plan)
    stages.report.run(manifest, edl, aligned, output, config)
    return 0


def load_storyboard(work: Path, slug: str) -> Storyboard:
    caminho = work / "storyboard.json"
    if not caminho.exists():
        raise SystemExit(
            f"storyboard.json nao existe em {work} — "
            f"rode antes: pipeline storyboard {slug}"
        )
    return read_json(caminho, Storyboard)


def cmd_approve(args, config: Config) -> int:
    """Aprova um portao, gravando O QUE foi aprovado.

    Guardar o digest e o que faz editar depois de aprovar revogar a aprovacao,
    em vez de o estagio seguinte rodar no texto novo com o aval do antigo.
    """
    work = config.work_dir / args.slug

    if args.gate == "script":
        script = read_script(work / "script.md")
        approval.grant(work, "script", script.digest())
        print(f"\n  Roteiro aprovado. Proximo:  pipeline storyboard {args.slug}\n")
        return 0

    if args.gate == "storyboard":
        # Le o artefato em disco em vez de chamar `stages.storyboard.run`.
        # Aprovar tem que aprovar O QUE VOCE LEU, e passar pelo estagio podia
        # gastar uma chamada de Opus se o config tivesse mudado no meio —
        # aprovar nao e hora de gastar. Aprovar um hash desatualizado tambem
        # nao deixa nada passar: o estagio seguinte recalcula a propria chave
        # de cache, e storyboard recalculado volta a precisar de aprovacao.
        caminho = work / "storyboard.json"
        if not caminho.exists():
            raise SystemExit(
                f"storyboard.json nao existe em {work} — "
                f"rode antes: pipeline storyboard {args.slug}"
            )
        board = read_json(caminho, Storyboard)
        approval.grant(work, "storyboard", board.input_hash)
        print(f"\n  Storyboard aprovado ({board.n_images} imagens). "
              f"Proximo:  pipeline images {args.slug}\n")
        return 0

    if args.gate == "images":
        work = config.work_dir / args.slug
        caminho = work / "assets.json"
        if not caminho.exists():
            raise SystemExit(
                f"assets.json nao existe em {work} — "
                f"rode antes: pipeline images {args.slug}"
            )
        assets = read_json(caminho, Assets)
        approval.grant(work, "images", assets.digest())
        print(f"\n  {len(assets.items)} imagens aprovadas "
              f"(USD {assets.total_cost_usd:.4f}). Agora grave, e depois:"
              f"\n    pipeline shoot {args.slug} <arquivo.mp4>\n")
        return 0

    raise SystemExit(f"portao desconhecido: {args.gate}")


def cmd_status(args, config: Config) -> int:
    """Onde cada video esta. A mesma leitura que a pagina inicial da UI faz."""
    estados = progress.all_states(config)
    if not estados:
        print("\n  Nenhum video em work/ ainda.\n")
        return 0

    # A largura sai do slug mais longo, nao de um numero fixo: o slug vem das
    # seis primeiras palavras da ideia (`stages.script.slug_for`) e passa de 34
    # com frequencia — e ai a coluna fixa colava uma na outra.
    largura = max(len("video"), *(len(e.slug) for e in estados))

    print()
    print(f"  {'video':<{largura}} {'roteiro':<11} {'storyboard':<11} "
          f"{'imagens':<11} {'gravacao'}")
    print("  " + "-" * (largura + 46))
    for estado in estados:
        colunas = "".join(f"{g.state:<11} " for _, g in estado.gates)
        print(f"  {estado.slug:<{largura}} {colunas}"
              f"{'sim' if estado.recorded else '-'}")
        for _, gate in estado.gates:
            if gate.error:
                print(f"    ! {gate.error}")
    print()
    print("  'editado!' quer dizer que o arquivo mudou depois de aprovado —")
    print("  a aprovacao foi revogada e precisa ser dada de novo.")
    print()
    return 0


def cmd_ui(args, config: Config) -> int:
    """Sobe o servidor local de revisao.

    127.0.0.1 e nao 0.0.0.0 de proposito: esta UI aprova gasto e escreve
    arquivo, sem autenticacao nenhuma. Ela e uma ferramenta de uma pessoa na
    propria maquina, e o default nao pode expor isso para a rede.
    """
    from .ui import serve

    serve(config, host=args.host, port=args.port)
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

    script_cmd = sub.add_parser("script", help="escreve o roteiro a partir de uma ideia")
    script_cmd.add_argument("idea", help="a ideia do video, nas suas palavras")
    script_cmd.add_argument("--seconds", type=float, default=900.0,
                            help="duracao alvo falada (default: 900 = 15 min)")
    script_cmd.add_argument("--slug", help="default: derivado da ideia")

    storyboard_cmd = sub.add_parser("storyboard",
                                    help="do roteiro aprovado para os beats visuais")
    storyboard_cmd.add_argument("slug")

    images_cmd = sub.add_parser("images",
                                help="do storyboard aprovado para as imagens")
    images_cmd.add_argument("slug")
    images_cmd.add_argument("--dry-run", action="store_true",
                            help="mostra a divisao banco/stock/geracao e o custo, sem gastar")

    approve_cmd = sub.add_parser("approve", help="aprova um portao")
    approve_cmd.add_argument("gate", choices=stages.GATES)
    approve_cmd.add_argument("slug")

    shoot_cmd = sub.add_parser(
        "shoot", help="depois de gravar: do arquivo ao video montado")
    shoot_cmd.add_argument("slug")
    shoot_cmd.add_argument("input", help="o arquivo de video que voce gravou")

    align_cmd = sub.add_parser("align", help="onde cada imagem cai na fala gravada")
    align_cmd.add_argument("slug")

    regen_cmd = sub.add_parser(
        "regenerate", help="gera outra imagem para um beat (gasta)")
    regen_cmd.add_argument("slug")
    regen_cmd.add_argument("beat", type=int, help="o beat_id da imagem a trocar")

    sub.add_parser("status", help="onde cada video esta")

    ui_cmd = sub.add_parser("ui", help="sobe a UI local de revisao e aprovacao")
    ui_cmd.add_argument("--host", default="127.0.0.1")
    ui_cmd.add_argument("--port", type=int, default=8000)

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
        if args.command == "script":
            return cmd_script(args, config)
        if args.command == "storyboard":
            return cmd_storyboard(args, config)
        if args.command == "images":
            return cmd_images(args, config)
        if args.command == "shoot":
            return cmd_shoot(args, config)
        if args.command == "align":
            return cmd_align(args, config)
        if args.command == "regenerate":
            return cmd_regenerate(args, config)
        if args.command == "approve":
            return cmd_approve(args, config)
        if args.command == "status":
            return cmd_status(args, config)
        if args.command == "ui":
            return cmd_ui(args, config)
        return cmd_stage(args.command, args.slug, config)
    except approval.NotApproved as exc:
        # Portao fechado nao e defeito do programa: e a resposta certa, e a
        # mensagem ja diz o comando exato. Deixar propagar imprimia um
        # traceback de Python em cima de um recado escrito para uma pessoa.
        raise SystemExit(f"\n  {exc}\n") from None
    except KeyboardInterrupt:
        log("interrupted", detail="os artefatos ja escritos continuam validos; rode de novo para retomar")
        return 130


if __name__ == "__main__":
    sys.exit(main())
