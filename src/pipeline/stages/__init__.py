"""Os seis estagios. Cada um le e escreve artefatos em work/<slug>/.

Os modulos sao expostos, nao as funcoes: `stages.render.run(...)` deixa claro
de qual estagio a funcao e, e evita que o nome do modulo e o da funcao se
sombreiem.
"""

from . import (
    align, assets, images, ingest, plan, render, report, script, storyboard,
    transcribe, trim,
)

# A ordem do pipeline de gravacao-primeiro, que continua funcionando inteiro.
ORDER = ["ingest", "transcribe", "trim", "plan", "assets", "render", "report"]

# A ordem do pipeline de roteiro-primeiro. Os TRES primeiros rodam antes de a
# camera ligar, cada um atras de um portao de aprovacao; os de baixo rodam
# sozinhos depois da gravacao (`pipeline shoot`). Ver
# docs/plans/2026-09-16-roteiro-primeiro.md.
SCRIPT_FIRST = [
    "script", "storyboard", "images",
    "ingest", "transcribe", "trim", "align", "render", "report",
]

# Os portoes, na ordem, com o artefato que cada um cobre. A UI e o `status`
# leem daqui para nao existir uma segunda lista de portoes em outro lugar.
GATES = ["script", "storyboard", "images"]

__all__ = [
    "ORDER", "SCRIPT_FIRST", "GATES",
    "script", "storyboard", "images", "align",
    "ingest", "transcribe", "trim", "plan", "assets", "render", "report",
]
