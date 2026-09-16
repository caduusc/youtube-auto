"""Estagio 2: o roteiro aprovado vira beats visuais.

Exige a aprovacao do roteiro, e a exigencia nao e burocratica: o storyboard
gasta uma chamada de Opus, e rodar em cima de um roteiro que voce ainda vai
reescrever joga aquela chamada fora.
"""

from __future__ import annotations

import anthropic

from .. import approval
from ..config import Config, require_key
from ..log import log, stage
from ..schemas import Script, Storyboard
from ..script import spoken_seconds
from ..storyboard import plan as plan_storyboard
from ..util import read_json_if_fresh, text_hash, write_json


def cache_key(script: Script, config: Config) -> str:
    """O digest do roteiro mais tudo que muda o que o storyboard decide.

    As regras editoriais e a duracao do sub-plano entram porque mudam a
    cobertura pedida no prompt; o estilo entra porque muda o `concept`. Sem
    isso, baixar `sub_shot_seconds` no config nao invalidaria um storyboard
    calculado para a duracao antiga.
    """
    return text_hash(
        script.digest(),
        config.editorial.model_dump_json(),
        f"{config.render.sub_shot_seconds:.3f}",
        f"{config.render.max_sub_shots}",
        f"{config.script.words_per_minute:.3f}",
        config.style_name,
    )


def run(
    script: Script,
    *,
    config: Config,
    client: anthropic.Anthropic | None = None,
) -> Storyboard:
    work = config.work_dir / script.slug
    caminho = work / "storyboard.json"
    key = cache_key(script, config)

    approval.require(
        work, "script", script.digest(),
        what="O roteiro",
        command=f"pipeline approve script {script.slug}",
    )

    cached = read_json_if_fresh(caminho, Storyboard, key)
    if cached is not None:
        log("storyboard.cached", slug=script.slug, imagens=cached.n_images)
        return cached

    falado = spoken_seconds(script, config.script.words_per_minute)

    with stage("storyboard", slug=script.slug, falado=f"{falado:.0f}s"):
        if client is None:
            client = anthropic.Anthropic(
                api_key=require_key(config.anthropic.env, "storyboard")
            )

        board = plan_storyboard(
            script, spoken_seconds=falado, input_hash=key,
            config=config.anthropic, rules=config.editorial,
            render=config.render, client=client,
        )
        write_json(caminho, board)
        return board
