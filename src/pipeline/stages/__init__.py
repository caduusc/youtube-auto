"""Os seis estagios. Cada um le e escreve artefatos em work/<slug>/.

Os modulos sao expostos, nao as funcoes: `stages.render.run(...)` deixa claro
de qual estagio a funcao e, e evita que o nome do modulo e o da funcao se
sombreiem.
"""

from . import assets, ingest, plan, render, report, transcribe

ORDER = ["ingest", "transcribe", "plan", "assets", "render", "report"]

__all__ = ["ORDER", "ingest", "transcribe", "plan", "assets", "render", "report"]
