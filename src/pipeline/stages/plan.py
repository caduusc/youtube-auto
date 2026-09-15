"""Estagio 3: transcript -> EDL, validada contra as regras editoriais."""

from __future__ import annotations

from ..config import Config
from ..log import log, stage
from ..planner import plan as call_model
from ..schemas import EDL, Manifest, Transcript
from ..util import read_json_if_fresh, write_json


def run(manifest: Manifest, transcript: Transcript, config: Config) -> EDL:
    work = config.work_dir / manifest.slug
    edl_path = work / "edl.json"

    cached = read_json_if_fresh(edl_path, EDL, transcript.digest())
    if cached is not None:
        log("plan.cached", slug=manifest.slug, n_broll=cached.stats.n_broll)
        return cached

    with stage("plan", model=config.anthropic.model, segments=len(transcript.segments)):
        edl = call_model(transcript, config=config.anthropic, rules=config.editorial)
        write_json(edl_path, edl)
        return edl
