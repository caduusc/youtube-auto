"""Provider de imagem plugavel e busca de stock gratuito.

`ImageProvider` e o contrato. Trocar de provider e escrever outra classe com
esses dois metodos e apontar `image_provider.active` no config para ela.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import requests

from .config import NetworkConfig, ReplicateConfig, StockConfig, require_key
from .log import log
from .util import with_retries


class ImageProvider(Protocol):
    @property
    def cost_usd_per_image(self) -> float: ...

    def generate(self, prompt: str, destination: Path) -> float:
        """Gera uma imagem em `destination`. Devolve o custo em USD."""


def _download(
    url: str, destination: Path, *, timeout: float, network: NetworkConfig,
    headers: dict | None = None,
) -> None:
    """Baixa para um temporario e so move ao completar.

    Escrever direto no destino deixaria um arquivo truncado no lugar se a
    conexao caisse no meio — e o resto do pipeline trataria aquilo como uma
    imagem valida.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    def attempt() -> None:
        response = requests.get(url, timeout=timeout, headers=headers or {}, stream=True)
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=65536):
                handle.write(chunk)
        partial.replace(destination)

    try:
        with_retries(
            attempt,
            attempts=network.download_attempts,
            base_delay=network.base_delay_seconds,
            label=f"download {destination.name}",
        )
    finally:
        partial.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Replicate / FLUX
# --------------------------------------------------------------------------


class ReplicateProvider:
    """FLUX via Replicate. `Prefer: wait` resolve a maioria em uma chamada."""

    BASE = "https://api.replicate.com/v1"

    def __init__(self, config: ReplicateConfig, network: NetworkConfig) -> None:
        self.config = config
        self.network = network
        self.token = require_key(config.env, "provider de imagem (replicate)")

    @property
    def cost_usd_per_image(self) -> float:
        return self.config.cost_usd_per_image

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def generate(self, prompt: str, destination: Path) -> float:
        try:
            # `api_attempts` e nao `download_attempts`: criar predicao cobra,
            # e uma resposta perdida repetida gera imagem duplicada.
            prediction = with_retries(
                lambda: self._create(prompt),
                attempts=self.network.api_attempts,
                base_delay=self.network.base_delay_seconds,
                label=f"replicate {self.config.model}",
            )
        except RuntimeError as exc:
            cause = exc.__cause__
            if isinstance(cause, requests.HTTPError):
                explained = self._explain_http_error(cause)
                if explained:
                    raise RuntimeError(explained) from cause
            raise
        url = self._await_output(prediction)
        _download(url, destination, timeout=self.config.timeout_seconds,
                  network=self.network)
        log("provider.generated", model=self.config.model, file=destination.name,
            cost=f"${self.cost_usd_per_image:.4f}")
        return self.cost_usd_per_image

    @staticmethod
    def _explain_http_error(exc: requests.HTTPError) -> str | None:
        """Erro de conta nao e problema de rede; retentar nao ajuda."""
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403):
            return (
                "o Replicate recusou a credencial (HTTP {}). Confira o valor de "
                "REPLICATE_API_TOKEN.".format(status)
            )
        if status == 402:
            return (
                "o Replicate respondeu HTTP 402 (sem credito). Adicione credito "
                "na conta, ou rode com `image_provider.active: none` no config "
                "para usar so banco e stock, com cor solida no que sobrar."
            )
        return None

    def _create(self, prompt: str) -> dict:
        response = requests.post(
            f"{self.BASE}/models/{self.config.model}/predictions",
            headers={**self._headers, "Prefer": "wait"},
            # O config manda o `input` inteiro; o pipeline acrescenta o que e
            # dele. `prompt` por ultimo de proposito: nenhuma chave do YAML
            # pode sobrescrever o prompt que o storyboard escreveu.
            json={
                "input": {
                    **self.config.input,
                    "output_format": self.config.output_format,
                    "prompt": prompt,
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


def build_provider(active: str, config, network: NetworkConfig) -> ImageProvider | None:
    """Constroi o provider, ou None quando geracao esta desligada.

    `active: none` nao e um erro: e a escolha de rodar so com banco e stock,
    sem conta de geracao. Os segmentos que nenhum dos dois resolve caem no
    fallback de cor solida em vez de derrubar o estagio.
    """
    if active in {"none", "off", "disabled"}:
        return None
    if active == "replicate":
        return ReplicateProvider(config.replicate, network)
    raise RuntimeError(
        f"image_provider.active='{active}' nao tem implementacao; "
        "use 'replicate' ou 'none'"
    )


# --------------------------------------------------------------------------
# Pexels
# --------------------------------------------------------------------------


class PexelsStock:
    BASE = "https://api.pexels.com/v1/search"

    def __init__(self, config: StockConfig, network: NetworkConfig) -> None:
        self.config = config
        self.network = network
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

        payload = with_retries(
            attempt, attempts=self.network.api_attempts,
            base_delay=self.network.base_delay_seconds, label=f"pexels '{query}'",
        )
        for photo in payload.get("photos", []):
            if photo.get("width", 0) >= self.config.min_width:
                url = photo.get("src", {}).get("original")
                if url:
                    return url, f"Pexels / {photo.get('photographer', 'unknown')}"
        log("stock.miss", query=query, results=len(payload.get("photos", [])))
        return None

    def download(self, url: str, destination: Path) -> None:
        _download(url, destination, timeout=self.config.timeout_seconds,
                  network=self.network)
