"""Embeddings locais. O import pesado e lazy de proposito.

sentence-transformers arrasta o torch (centenas de MB). Deixar o import
dentro do metodo permite rodar a suite de testes, o `bank stats` e o
`--dry-run` sem ele carregado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .log import log


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Carrega de um ID do HuggingFace ou de uma pasta local.

    `model_name` e `load_path` sao coisas diferentes de proposito:

    * `model_name` e a IDENTIDADE do espaco vetorial, guardada em cada linha
      do banco para nunca comparar embeddings de modelos diferentes. Precisa
      ser estavel entre maquinas, entao e sempre o valor do config —
      `models/all-MiniLM-L6-v2`, nao `C:\\Users\\voce\\proj\\models\\...`.
    * `load_path` e o que vai para o `SentenceTransformer`. Para uma pasta
      local ele e absoluto, resolvido contra a raiz do projeto, para o
      pipeline funcionar rodado de qualquer diretorio.
    """

    def __init__(self, model_name: str, load_path: str | Path | None = None) -> None:
        self.model_name = model_name
        self.load_path = str(load_path) if load_path is not None else model_name
        self._model = None

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log("embed.load", model=self.model_name,
                **({"from": self.load_path} if self.load_path != self.model_name else {}))
            self._model = SentenceTransformer(self.load_path)
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vector]


def resolve_model(configured: str, root: Path) -> tuple[str, str]:
    """Devolve (identidade, caminho de carga) para o valor do config.

    Uma pasta existente sob a raiz do projeto vence: e exatamente o caso que
    o proprio HuggingFace avisa ("make sure you don't have a local directory
    with the same name"), e aqui ela e intencional — o escape para quem nao
    consegue baixar do Hub.
    """
    candidate = Path(configured)
    local = candidate if candidate.is_absolute() else root / candidate
    if local.is_dir():
        return configured, str(local.resolve())
    return configured, configured
