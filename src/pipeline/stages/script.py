"""Estagio 1: a ideia vira roteiro.

Nao toca em midia e nao depende de gravacao nenhuma — roda antes de a camera
ligar. O artefato e Markdown para voce editar a mao; ver `script.py`.
"""

from __future__ import annotations

import anthropic

from ..config import Config, require_key
from ..log import log, stage
from ..schemas import Script
from ..script import read as read_script
from ..script import spoken_seconds, words, write as write_script
from ..util import slugify
from ..writer import write as write_roteiro


def run(
    idea: str,
    *,
    slug: str,
    target_seconds: float,
    config: Config,
    client: anthropic.Anthropic | None = None,
) -> Script:
    work = config.work_dir / slug
    caminho = work / "script.md"

    # Cache por ARQUIVO e nao por hash da ideia, de proposito: depois de o
    # roteiro existir, a fonte da verdade e o arquivo que voce editou, nao o
    # texto da ideia original. Reescrever por cima apagaria a sua edicao.
    if caminho.exists():
        existente = read_script(caminho)
        log("script.cached", slug=slug, beats=len(existente.beats),
            palavras=words(existente),
            detail="ja existe; apague o arquivo para reescrever do zero")
        return existente

    with stage("script", slug=slug, alvo=f"{target_seconds:.0f}s"):
        if client is None:
            client = anthropic.Anthropic(
                api_key=require_key(config.anthropic.env, "roteiro")
            )

        script = write_roteiro(
            idea, slug=slug, target_seconds=target_seconds,
            config=config.anthropic, rules=config.script, client=client,
        )
        write_script(caminho, script)
        log("script.written", file=str(caminho),
            estimado=f"{spoken_seconds(script, config.script.words_per_minute):.0f}s")
        return script


def slug_for(idea: str) -> str:
    """Slug a partir das primeiras palavras da ideia.

    O roteiro nasce antes do arquivo de video, entao nao ha nome de arquivo de
    onde tirar o slug como o `ingest` faz.
    """
    return slugify(" ".join(idea.split()[:6]))
