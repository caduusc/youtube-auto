"""Embeddings locais. O import pesado e lazy de proposito.

sentence-transformers arrasta o torch (centenas de MB). Deixar o import
dentro do metodo permite rodar a suite de testes, o `bank stats` e o
`--dry-run` sem ele carregado.
"""

from __future__ import annotations

from typing import Protocol

from .log import log


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log("embed.load", model=self.model_name)
            self._model = SentenceTransformer(self.model_name)
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vector]
