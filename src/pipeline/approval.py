"""Portoes de aprovacao, em disco.

A aprovacao e um arquivo e nao estado de processo por duas razoes. A UI nao
pode ser a dona do estado — voce tem que poder fechar o navegador, ou rodar
tudo pela CLI sem UI nenhuma. E o pipeline ja e resumivel por artefato em
disco, entao um portao e so mais um artefato.

O arquivo guarda O QUE foi aprovado, nao apenas que houve aprovacao. Sem isso
existe um jeito silencioso de errar: voce aprova o roteiro, edita o texto, e o
estagio seguinte roda no texto novo carregando a aprovacao do antigo. Guardando
o digest, editar depois de aprovar REVOGA a aprovacao — e o erro aparece antes
de a imagem ser gerada, nao depois.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .log import log
from .util import now_iso


@dataclass
class Approval:
    """O que o arquivo de aprovacao guarda."""

    digest: str
    approved_at: str

    def matches(self, digest: str) -> bool:
        return self.digest == digest


class NotApproved(RuntimeError):
    """Mensagem sempre diz o que fazer: quem le isso e uma pessoa."""


def path_for(work: Path, gate: str) -> Path:
    return work / f"{gate}.approved"


def read(work: Path, gate: str) -> Approval | None:
    caminho = path_for(work, gate)
    if not caminho.exists():
        return None
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        return Approval(digest=str(dados["digest"]), approved_at=str(dados["approved_at"]))
    except (json.JSONDecodeError, KeyError, TypeError):
        # Arquivo corrompido vale como ausencia de aprovacao: o caminho seguro
        # e pedir de novo, nao adivinhar que estava aprovado.
        log("approval.corrupt", gate=gate, file=str(caminho))
        return None


def grant(work: Path, gate: str, digest: str) -> Approval:
    approval = Approval(digest=digest, approved_at=now_iso())
    caminho = path_for(work, gate)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps({"digest": approval.digest, "approved_at": approval.approved_at},
                   indent=2) + "\n",
        encoding="utf-8",
    )
    log("approval.granted", gate=gate, digest=digest[:12])
    return approval


def revoke(work: Path, gate: str) -> bool:
    caminho = path_for(work, gate)
    if not caminho.exists():
        return False
    caminho.unlink()
    log("approval.revoked", gate=gate)
    return True


def require(work: Path, gate: str, digest: str, *, what: str, command: str) -> None:
    """Deixa passar, ou explica em portugues o que fazer.

    `what` e `command` existem para a mensagem ser acionavel. "nao aprovado"
    manda a pessoa procurar; "revise work/x/script.md e rode `pipeline approve
    script x`" ela resolve.
    """
    approval = read(work, gate)

    if approval is None:
        raise NotApproved(
            f"{what} ainda nao foi aprovado.\n"
            f"Revise e aprove com:  {command}"
        )

    if not approval.matches(digest):
        raise NotApproved(
            f"{what} mudou depois de aprovado — a aprovacao de "
            f"{approval.approved_at} valia para outra versao.\n"
            f"Revise as mudancas e aprove de novo com:  {command}"
        )
