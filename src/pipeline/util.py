"""Hashing, slugs e leitura/escrita dos artefatos JSON."""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Arquivos de video sao grandes; ler tudo para hashear custa minutos de I/O.
# Amostrar inicio, meio e fim mais o tamanho identifica o conteudo bem o
# suficiente para um cache local.
_SAMPLE = 4 * 1024 * 1024


def file_hash(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as handle:
        for offset in (0, max(0, size // 2 - _SAMPLE // 2), max(0, size - _SAMPLE)):
            handle.seek(offset)
            digest.update(handle.read(_SAMPLE))
    return digest.hexdigest()


def text_hash(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()


def slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug or "video"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, model_cls: type[T]) -> T:
    return model_cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


def read_json_if_fresh(path: Path, model_cls: type[T], input_hash: str) -> T | None:
    """Devolve o artefato em disco se ele foi produzido a partir deste input.

    E isso que torna cada estagio idempotente e o pipeline resumivel: o
    artefato carrega o hash da propria entrada, entao o estagio sabe sozinho
    se o trabalho dele ja esta feito.
    """
    if not path.exists():
        return None
    try:
        artifact = read_json(path, model_cls)
    except Exception:
        return None
    return artifact if getattr(artifact, "input_hash", None) == input_hash else None


def with_retries(
    call: Callable[[], Any],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    label: str = "request",
) -> Any:
    """3 retries com backoff exponencial para erro de rede em chamada de API."""
    from .log import log

    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # a distincao retryable/fatal fica no chamador
            last = exc
            if attempt == attempts:
                break
            delay = base_delay * 2 ** (attempt - 1)
            log("retry", label=label, attempt=attempt, of=attempts, sleep=f"{delay:.0f}s", error=repr(exc))
            time.sleep(delay)
    raise RuntimeError(f"{label} falhou depois de {attempts} tentativas") from last
