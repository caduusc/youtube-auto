"""Estagio 6: o relatorio que diz se a conta fechou."""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..ffmpeg import probe
from ..log import stage
from ..schemas import EDL, Assets, Manifest, Report
from ..util import now_iso, write_json


def build(manifest: Manifest, edl: EDL, assets: Assets, output: Path, config: Config) -> Report:
    duration = float(probe(output)["format"]["duration"]) if output.exists() else edl.duration

    n_broll = edl.stats.n_broll
    free = sum(1 for i in assets.items if i.origin in {"bank", "stock"})
    cost_usd = assets.total_cost_usd
    minutes = duration / 60.0 if duration else 0.0

    return Report(
        input_hash=assets.input_hash,
        slug=manifest.slug,
        output_path=str(output),
        duration_seconds=round(duration, 3),
        n_aroll=edl.stats.n_aroll,
        n_broll=n_broll,
        broll_ratio=round(edl.stats.broll_ratio, 4),
        reuse_rate=round(free / n_broll, 4) if n_broll else 0.0,
        by_origin=assets.by_origin,
        cost_usd=round(cost_usd, 4),
        cost_brl=round(cost_usd * config.budget.brl_per_usd, 2),
        brl_per_usd=config.budget.brl_per_usd,
        cost_usd_per_final_minute=round(cost_usd / minutes, 4) if minutes else 0.0,
        generated_at=now_iso(),
    )


def render_text(report: Report) -> str:
    origins = ", ".join(f"{k}={v}" for k, v in sorted(report.by_origin.items())) or "nenhum"
    minutes, seconds = divmod(report.duration_seconds, 60)
    return "\n".join([
        "",
        f"  {report.slug}",
        "  " + "=" * 52,
        f"  duracao final              {int(minutes)}m{int(seconds):02d}s",
        f"  segmentos                  {report.n_aroll} a-roll / {report.n_broll} b-roll",
        f"  tela em b-roll             {report.broll_ratio:.0%}",
        f"  origem das imagens         {origins}",
        "  " + "-" * 52,
        f"  reuso sem custo            {report.reuse_rate:.0%} dos b-rolls",
        f"  custo total                USD {report.cost_usd:.4f}   "
        f"BRL {report.cost_brl:.2f}   (a {report.brl_per_usd:.2f}/USD)",
        f"  custo por minuto final     USD {report.cost_usd_per_final_minute:.4f}   "
        f"BRL {report.cost_usd_per_final_minute * report.brl_per_usd:.2f}",
        "  " + "=" * 52,
        f"  {report.output_path}",
        "",
    ])


def run(manifest: Manifest, edl: EDL, assets: Assets, output: Path, config: Config) -> Report:
    with stage("report"):
        report = build(manifest, edl, assets, output, config)
        write_json(config.work_dir / manifest.slug / "report.json", report)
        print(render_text(report))
        return report
