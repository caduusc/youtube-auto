"""Provider de imagem plugavel e busca de stock gratuito.

`ImageProvider` e o contrato. Trocar de provider e escrever outra classe com
esses dois metodos e apontar `image_provider.active` no config para ela.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import requests

from .config import ReplicateConfig, StockConfig, require_key
from .log import log
from .util import with_retries


class ImageProvider(Protocol):
    @property
    def cost_usd_per_image(self) -> float: ...

    def generate(self, prompt: str, destination: Path) -> float:
        """Gera uma imagem em `destination`. Devolve o custo em USD."""


def _download(url: str, destination: Path, *, timeout: float, headers: dict | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    def attempt() -> None:
        response = requests.get(url, timeout=timeout, headers=headers or {}, stream=True)
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=65536):
                handle.write(chunk)

    with_retries(attempt, label=f"download {destination.name}")


# --------------------------------------------------------------------------
# Replicate / FLUX
# --------------------------------------------------------------------------


class ReplicateProvider:
    """FLUX via Replicate. `Prefer: wait` resolve a maioria em uma chamada."""

    BASE = "https://api.replicate.com/v1"

    def __init__(self, config: ReplicateConfig) -> None:
        self.config = config
        self.token = require_key(config.env, "provider de imagem (replicate)")

    @property
    def cost_usd_per_image(self) -> float:
        return self.config.cost_usd_per_image

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def generate(self, prompt: str, destination: Path) -> float:
        prediction = with_retries(
            lambda: self._create(prompt), label=f"replicate {self.config.model}"
        )
        url = self._await_output(prediction)
        _download(url, destination, timeout=self.config.timeout_seconds)
        log("provider.generated", model=self.config.model, file=destination.name,
            cost=f"${self.cost_usd_per_image:.4f}")
        return self.cost_usd_per_image

    def _create(self, prompt: str) -> dict:
        response = requests.post(
            f"{self.BASE}/models/{self.config.model}/predictions",
            headers={**self._headers, "Prefer": "wait"},
            json={
                "input": {
                    "prompt": prompt,
                    "aspect_ratio": self.config.aspect_ratio,
                    "output_format": self.config.output_format,
                    "num_outputs": 1,
                }
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _await_output(self, prediction: dict) -> str:
        """`Prefer: wait` segura ate ~60s; alem disso, poll."""
        deadline = time.monotonic() + self.config.timeout_seconds
        while prediction.get("status") in {"starting", "processing"}:
            if time.monotonic() > deadline:
                raise RuntimeError(f"replicate nao terminou em {self.config.timeout_seconds:.0f}s")
            time.sleep(2.0)
            poll_url = prediction.get("urls", {}).get("get")
            if not poll_url:
                raise RuntimeError("replicate nao devolveu urls.get para continuar o poll")
            response = requests.get(poll_url, headers=self._headers, timeout=30)
            response.raise_for_status()
            prediction = response.json()

        if prediction.get("status") != "succeeded":
            raise RuntimeError(
                f"replicate status={prediction.get('status')} error={prediction.get('error')}"
            )
        output = prediction.get("output")
        if isinstance(output, list):
            output = output[0] if output else None
        if not isinstance(output, str):
            raise RuntimeError(f"replicate devolveu output inesperado: {output!r}")
        return output


def build_provider(active: str, config) -> ImageProvider:
    if active == "replicate":
        return ReplicateProvider(config.replicate)
    raise RuntimeError(
        f"image_provider.active='{active}' nao tem implementacao; use 'replicate'"
    )


# --------------------------------------------------------------------------
# Pexels
# --------------------------------------------------------------------------


class PexelsStock:
    BASE = "https://api.pexels.com/v1/search"

    def __init__(self, config: StockConfig) -> None:
        self.config = config
        self.key = require_key(config.env, "stock (pexels)")

    def is_generic(self, tags: list[str]) -> bool:
        """Roteia para stock se qualquer tag casar com a lista do config."""
        haystack = " ".join(tags).lower()
        return any(generic.lower() in haystack for generic in self.config.generic_tags)

    def search(self, tags: list[str]) -> tuple[str, str] | None:
        """Devolve (url, credito) da primeira foto larga o suficiente."""
        query = " ".join(tags)

        def attempt() -> dict:
            response = requests.get(
                self.BASE,
                headers={"Authorization": self.key},
                params={
                    "query": query,
                    "orientation": self.config.orientation,
                    "per_page": 15,
                },
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()

        payload = with_retries(attempt, label=f"pexels '{query}'")
        for photo in payload.get("photos", []):
            if photo.get("width", 0) >= self.config.min_width:
                url = photo.get("src", {}).get("original")
                if url:
                    return url, f"Pexels / {photo.get('photographer', 'unknown')}"
        log("stock.miss", query=query, results=len(payload.get("photos", [])))
        return None

    def download(self, url: str, destination: Path) -> None:
        _download(url, destination, timeout=self.config.timeout_seconds)
