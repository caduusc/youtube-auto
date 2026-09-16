"""Os seis estagios. Cada um le e escreve artefatos em work/<slug>/.

Os modulos sao expostos, nao as funcoes: `stages.render.run(...)` deixa claro
de qual estagio a funcao e, e evita que o nome do modulo e o da funcao se
sombreiem.
"""

from . import assets, ingest, plan, render, report, script, storyboard, transcribe, trim

# A ordem do pipeline de gravacao-primeiro, que continua funcionando inteiro.
ORDER = ["ingest", "transcribe", "trim", "plan", "assets", "render", "report"]

# A ordem do pipeline de roteiro-primeiro. Os tres primeiros rodam ANTES de a
# camera ligar, cada um atras de um portao de aprovacao; os de baixo rodam
# sozinhos depois da gravacao. Ver docs/plans/2026-09-16-roteiro-primeiro.md.
SCRIPT_FIRST = ["script", "storyboard", "ingest", "transcribe", "render"]

__all__ = [
    "ORDER", "SCRIPT_FIRST",
    "script", "storyboard",
    "ingest", "transcribe", "trim", "plan", "assets", "render", "report",
]
