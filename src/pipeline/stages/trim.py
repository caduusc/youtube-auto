"""Estagio 2b: corte seco de pausa longa e hesitacao.

Este estagio nao toca em midia. Ele calcula o plano de corte e devolve o
transcript remapeado para a timeline nova; o corte propriamente dito acontece
no estagio 5, dentro do mesmo filtergraph do overlay.

A razao e uma so: produzir um arquivo cortado aqui custaria um encode de
video a mais, e o estagio 5 encodaria de novo em cima. Dois encodes num video
de 15 min sao quase duas horas de CPU e uma perda de qualidade que nao
precisa existir.
"""

from __future__ import annotations

from ..config import Config
from ..log import log, stage
from ..schemas import Manifest, Transcript, TrimPlan
from ..trim import build_plan, remap_transcript, resolve_fps
from ..util import read_json_if_fresh, write_json


def run(manifest: Manifest, transcript: Transcript, config: Config) -> tuple[TrimPlan, Transcript]:
    work = config.work_dir / manifest.slug
    plan_path = work / "trim.json"
    trimmed_path = work / "transcript.trimmed.json"

    fps = resolve_fps(config, manifest.fps)
    key = build_plan(transcript, config.trim, fps).input_hash

    cached = read_json_if_fresh(plan_path, TrimPlan, key)
    if cached is not None and trimmed_path.exists():
        log("trim.cached", slug=manifest.slug, cuts=cached.stats.n_cuts)
        from ..util import read_json

        return cached, read_json(trimmed_path, Transcript)

    with stage("trim", enabled=config.trim.enabled, fps=f"{fps:g}"):
        plan = build_plan(transcript, config.trim, fps)
        trimmed = remap_transcript(transcript, plan)

        write_json(plan_path, plan)
        write_json(trimmed_path, trimmed)

        if plan.enabled:
            log("trim.ok",
                cuts=plan.stats.n_cuts,
                pausas=plan.stats.n_pause_cuts,
                hesitacoes=plan.stats.n_filler_cuts,
                apertos=plan.stats.n_squeeze_cuts,
                removido=f"{plan.stats.removed_seconds:.1f}s",
                ratio=f"{plan.stats.removed_ratio:.1%}",
                duracao=f"{plan.stats.trimmed_seconds:.1f}s")
        else:
            log("trim.disabled", detail="corte desligado no config")

        return plan, trimmed


def render_text(plan: TrimPlan, limit: int = 40) -> str:
    """Os cortes, com o texto ao redor, para voce revisar antes de renderizar.

    Sem isso o corte e uma caixa preta: voce ve 12% de reducao e nao tem como
    saber se uma pausa retorica que dava peso a uma frase foi embora.
    """
    if not plan.enabled:
        return "\n  Corte desligado no config (trim.enabled: false).\n"
    if not plan.cuts:
        return "\n  Nenhum trecho morto encontrado com as regras atuais.\n"

    s = plan.stats
    lines = [
        "",
        "  Cortes",
        "  " + "-" * 72,
    ]
    for cut in plan.cuts[:limit]:
        marca = {"filler": "hesitacao", "squeeze": "aperto   "}.get(cut.reason, "pausa    ")
        lines.append(f"  {cut.start:7.1f}s  {marca}  {cut.duration:4.1f}s")
        lines.append(f"            {cut.context}")
    if len(plan.cuts) > limit:
        lines.append(f"  ... e outros {len(plan.cuts) - limit} cortes (ver trim.json)")
    lines += [
        "  " + "-" * 72,
        f"  {s.n_cuts} corte{'s' if s.n_cuts != 1 else ''}: "
        f"{s.n_pause_cuts} pausa{'s' if s.n_pause_cuts != 1 else ''}, "
        f"{s.n_filler_cuts} hesitac{'oes' if s.n_filler_cuts != 1 else 'ao'}, "
        f"{s.n_squeeze_cuts} aperto{'s' if s.n_squeeze_cuts != 1 else ''}",
        f"  {s.original_seconds:.1f}s -> {s.trimmed_seconds:.1f}s "
        f"({s.removed_seconds:.1f}s removidos, {s.removed_ratio:.1%})",
        "",
    ]
    return "\n".join(lines)
