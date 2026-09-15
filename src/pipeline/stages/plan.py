"""Estagio 3: transcript -> EDL, validada contra as regras editoriais."""

from __future__ import annotations

from ..config import Config
from ..log import log, stage
from ..planner import plan as call_model
from ..schemas import EDL, Manifest, Transcript
from ..util import read_json_if_fresh, text_hash, write_json


def cache_key(transcript: Transcript, config: Config) -> str:
    """Identidade da EDL: o transcript MAIS as regras que a produziram.

    Sem as regras na chave, afrouxar `editorial` e rodar de novo devolve a
    EDL antiga em silencio — exatamente no fluxo de calibragem em que se
    mexe nas regras justamente para obter uma EDL diferente.
    """
    return text_hash(transcript.digest(), config.editorial.model_dump_json())


def run(manifest: Manifest, transcript: Transcript, config: Config) -> EDL:
    work = config.work_dir / manifest.slug
    edl_path = work / "edl.json"
    key = cache_key(transcript, config)

    cached = read_json_if_fresh(edl_path, EDL, key)
    if cached is not None:
        log("plan.cached", slug=manifest.slug, n_broll=cached.stats.n_broll)
        return cached

    with stage("plan", model=config.anthropic.model, segments=len(transcript.segments)):
        edl = call_model(transcript, config=config.anthropic, rules=config.editorial)
        edl.input_hash = key
        write_json(edl_path, edl)
        return edl
