"""Onde cada video esta. Uma fonte so, para a CLI e para a UI.

Duplicar isto era o caminho natural — a CLI imprime uma tabela, a UI renderiza
uma pagina — e a divergencia seria do tipo que nao falha: a UI diria
"aprovado" e o `status` diria "editado!", ou o contrario, e quem estivesse
olhando a tela erraria com confianca. O estado de um portao e uma pergunta so,
respondida aqui.

Os quatro estados de um portao:

    -           o artefato nao existe ainda (ou nao da para ler)
    escrito     existe e nao foi aprovado
    aprovado    a aprovacao em disco vale para ESTA versao
    editado!    houve aprovacao, mas para outra versao

O ultimo e o que justifica guardar o digest em vez de so um marcador. Ver o
docstring de `approval.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import approval
from .config import Config
from .schemas import Storyboard
from .script import ScriptFormatError, read as read_script
from .util import read_json

ABSENT = "-"
WRITTEN = "escrito"
APPROVED = "aprovado"
EDITED = "editado!"


@dataclass
class Gate:
    """O estado de um portao de aprovacao."""

    state: str
    # Preenchido quando o artefato existe mas nao da para ler. E a mensagem
    # que a pessoa precisa para consertar o arquivo, entao ela viaja junto com
    # o estado em vez de virar so um "-" sem explicacao.
    error: str = ""

    @property
    def pending(self) -> bool:
        """Ha o que aprovar agora."""
        return self.state in (WRITTEN, EDITED)


@dataclass
class VideoState:
    slug: str
    script: Gate
    storyboard: Gate
    recorded: bool

    @property
    def stage(self) -> str:
        """Em que ponto do caminho este video esta, em uma palavra."""
        if self.script.state == ABSENT:
            return "sem roteiro"
        if self.script.state != APPROVED:
            return "roteiro"
        if self.storyboard.state == ABSENT:
            return "aguardando storyboard"
        if self.storyboard.state != APPROVED:
            return "storyboard"
        return "gravado" if self.recorded else "pronto para gravar"


def script_gate(work: Path) -> Gate:
    try:
        script = read_script(work / "script.md")
    except ScriptFormatError as exc:
        caminho = work / "script.md"
        if not caminho.exists():
            return Gate(ABSENT)
        return Gate(ABSENT, error=str(exc))
    return _compare(approval.read(work, "script"), script.digest())


def storyboard_gate(work: Path) -> Gate:
    """O estado do storyboard, comparando o digest e nao so a existencia.

    Comparar importa aqui pelo mesmo motivo do roteiro, por um caminho menos
    obvio: o `input_hash` do storyboard inclui as regras editoriais, o
    `sub_shot_seconds` e o estilo. Mexer no config invalida o storyboard, e sem
    comparar o digest esta tela diria "aprovado" para um artefato que o
    proximo estagio vai recalcular.
    """
    caminho = work / "storyboard.json"
    if not caminho.exists():
        return Gate(ABSENT)
    try:
        board = read_json(caminho, Storyboard)
    except Exception as exc:                       # noqa: BLE001 - ver docstring
        # Qualquer falha de leitura vale como "existe mas nao da para usar":
        # JSON truncado, schema antigo, campo que mudou de nome. O que a
        # pessoa precisa saber e que este arquivo precisa ser refeito.
        return Gate(ABSENT, error=f"storyboard.json nao pode ser lido: {exc}")
    return _compare(approval.read(work, "storyboard"), board.input_hash)


def _compare(granted: approval.Approval | None, digest: str) -> Gate:
    if granted is None:
        return Gate(WRITTEN)
    return Gate(APPROVED if granted.matches(digest) else EDITED)


def state_of(work: Path) -> VideoState:
    return VideoState(
        slug=work.name,
        script=script_gate(work),
        storyboard=storyboard_gate(work),
        recorded=(work / "manifest.json").exists(),
    )


def all_states(config: Config) -> list[VideoState]:
    raiz = config.work_dir
    if not raiz.exists():
        return []
    return [state_of(p) for p in sorted(raiz.iterdir()) if p.is_dir()]
