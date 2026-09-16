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
from .schemas import Assets, Storyboard
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
    images: Gate
    recorded: bool

    @property
    def gates(self) -> list[tuple[str, Gate]]:
        return [("script", self.script), ("storyboard", self.storyboard),
                ("images", self.images)]

    @property
    def stage(self) -> str:
        """Em que ponto do caminho este video esta, em poucas palavras."""
        for nome, gate in self.gates:
            if gate.state == ABSENT:
                return "sem roteiro" if nome == "script" else f"aguardando {nome}"
            if gate.state != APPROVED:
                return nome
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


def images_gate(work: Path) -> Gate:
    """O estado das imagens.

    Mesmo desenho do storyboard, e pela mesma razao: o `input_hash` das
    imagens inclui o do storyboard mais o estilo e o provider, entao trocar o
    `style_suffix` no config invalida as imagens — e sem comparar o digest a
    tela diria "aprovado" para imagens que vao ser regeradas (e pagas de novo).

    `assets.json` existe nos DOIS caminhos, mas o portao e so do
    roteiro-primeiro: no gravacao-primeiro as imagens saem depois da EDL, sem
    nada para aprovar antes. O que distingue e a chave — item com `beat_id`
    veio de storyboard, item com `segment_index` veio de EDL. Sem esta
    distincao todo video antigo apareceria como "escrito" para sempre,
    esperando uma aprovacao que nao existe naquele caminho.
    """
    caminho = work / "assets.json"
    if not caminho.exists():
        return Gate(ABSENT)
    try:
        assets = read_json(caminho, Assets)
    except Exception as exc:                       # noqa: BLE001 - ver docstring
        return Gate(ABSENT, error=f"assets.json nao pode ser lido: {exc}")
    if not any(item.beat_id >= 0 for item in assets.items):
        return Gate(ABSENT)
    return _compare(approval.read(work, "images"), assets.input_hash)


def _compare(granted: approval.Approval | None, digest: str) -> Gate:
    if granted is None:
        return Gate(WRITTEN)
    return Gate(APPROVED if granted.matches(digest) else EDITED)


def state_of(work: Path) -> VideoState:
    return VideoState(
        slug=work.name,
        script=script_gate(work),
        storyboard=storyboard_gate(work),
        images=images_gate(work),
        recorded=(work / "manifest.json").exists(),
    )


def all_states(config: Config) -> list[VideoState]:
    raiz = config.work_dir
    if not raiz.exists():
        return []
    return [state_of(p) for p in sorted(raiz.iterdir()) if p.is_dir()]
