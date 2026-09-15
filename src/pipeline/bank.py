"""Banco de assets em SQLite, compartilhado entre todos os videos.

A similaridade de cosseno roda em Python puro sobre `array('f')`. Com alguns
milhares de assets sao ~400 mil multiplicacoes por busca — irrelevante ao
lado de uma chamada de rede — e em troca o banco fica utilizavel sem torch
nem numpy carregados.
"""

from __future__ import annotations

import math
import sqlite3
from array import array
from dataclasses import dataclass
from pathlib import Path

from .embed import Embedder
from .log import log
from .util import now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT    NOT NULL UNIQUE,
    concept       TEXT    NOT NULL,
    prompt        TEXT,
    origin        TEXT    NOT NULL CHECK (origin IN ('stock', 'generated')),
    embedding     BLOB    NOT NULL,
    embedding_dim INTEGER NOT NULL,
    cost_usd      REAL    NOT NULL DEFAULT 0.0,
    created_at    TEXT    NOT NULL,
    use_count     INTEGER NOT NULL DEFAULT 0,
    last_used_at  TEXT
);
CREATE INDEX IF NOT EXISTS assets_last_used ON assets (last_used_at);
"""


@dataclass
class Asset:
    id: int
    path: str
    concept: str
    prompt: str | None
    origin: str
    cost_usd: float
    created_at: str
    use_count: int
    last_used_at: str | None


def pack(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def unpack(blob: bytes) -> array:
    out = array("f")
    out.frombytes(blob)
    return out


def cosine(a: array | list[float], b: array | list[float]) -> float:
    """Cosseno. Nao assume que os vetores chegam normalizados."""
    if len(a) != len(b):
        return 0.0
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


class AssetBank:
    def __init__(
        self, root: Path, db_path: Path, images_dir: Path, embedder: Embedder, threshold: float
    ) -> None:
        # Os caminhos em `assets.path` sao guardados relativos a `root`, para o
        # banco continuar valido se o projeto mudar de lugar.
        self.root = Path(root)
        self.db_path = Path(db_path)
        self.images_dir = Path(images_dir)
        self.embedder = embedder
        self.threshold = threshold
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def absolute(self, stored_path: str) -> Path:
        candidate = Path(stored_path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def relative(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(Path(path).resolve())

    # -- leitura -----------------------------------------------------------

    def _row_to_asset(self, row: sqlite3.Row) -> Asset:
        return Asset(
            id=row["id"], path=row["path"], concept=row["concept"], prompt=row["prompt"],
            origin=row["origin"], cost_usd=row["cost_usd"], created_at=row["created_at"],
            use_count=row["use_count"], last_used_at=row["last_used_at"],
        )

    def find_similar(
        self, concept: str, *, exclude_ids: set[int] | None = None
    ) -> tuple[Asset, float] | None:
        """Melhor asset acima do limiar, ignorando os ja usados neste video.

        Reusar a mesma imagem duas vezes no mesmo video e pior que gerar uma
        nova — o espectador nota. Entre videos diferentes, reusar e o ponto.
        """
        exclude_ids = exclude_ids or set()

        # Banco vazio nao tem o que comparar, e o embedder arrasta o torch:
        # no primeiro video, e o que deixa o --dry-run responder na hora em
        # vez de esperar 400MB de modelo carregar para nada.
        if not self.conn.execute("SELECT 1 FROM assets LIMIT 1").fetchone():
            return None

        target = self.embedder.embed(concept)
        best: tuple[Asset, float] | None = None
        for row in self.conn.execute("SELECT * FROM assets"):
            if row["id"] in exclude_ids:
                continue
            if not self.absolute(row["path"]).exists():
                continue  # arquivo sumiu do disco; a linha fica para o prune
            score = cosine(target, unpack(row["embedding"]))
            if score >= self.threshold and (best is None or score > best[1]):
                best = (self._row_to_asset(row), score)
        return best

    # -- escrita -----------------------------------------------------------

    def add(
        self, *, path: str, concept: str, origin: str, prompt: str | None = None,
        cost_usd: float = 0.0, embedding: list[float] | None = None,
    ) -> Asset:
        vector = embedding if embedding is not None else self.embedder.embed(concept)
        cursor = self.conn.execute(
            """INSERT INTO assets (path, concept, prompt, origin, embedding, embedding_dim,
                                   cost_usd, created_at, use_count, last_used_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
            (path, concept, prompt, origin, pack(vector), len(vector), cost_usd, now_iso()),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM assets WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return self._row_to_asset(row)

    def mark_used(self, asset_id: int) -> None:
        self.conn.execute(
            "UPDATE assets SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
            (now_iso(), asset_id),
        )
        self.conn.commit()

    # -- manutencao --------------------------------------------------------

    def stats(self) -> dict:
        row = self.conn.execute(
            """SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd), 0) AS spent,
                      COALESCE(SUM(use_count), 0) AS uses
               FROM assets"""
        ).fetchone()
        by_origin = {
            r["origin"]: r["n"]
            for r in self.conn.execute("SELECT origin, COUNT(*) AS n FROM assets GROUP BY origin")
        }
        never = self.conn.execute("SELECT COUNT(*) AS n FROM assets WHERE use_count = 0").fetchone()["n"]
        return {
            "assets": row["n"],
            "by_origin": by_origin,
            "total_spent_usd": round(row["spent"], 4),
            "total_uses": row["uses"],
            "never_used": never,
            "avg_uses_per_asset": round(row["uses"] / row["n"], 2) if row["n"] else 0.0,
            "db_path": str(self.db_path),
        }

    def prune(self, unused_days: int, *, delete_files: bool = True) -> list[str]:
        """Remove assets sem uso ha N dias (ou nunca usados e criados ha N dias)."""
        cutoff = f"-{unused_days} days"
        rows = self.conn.execute(
            """SELECT * FROM assets
               WHERE COALESCE(last_used_at, created_at) < datetime('now', ?)""",
            (cutoff,),
        ).fetchall()
        removed: list[str] = []
        for row in rows:
            if delete_files:
                target = self.absolute(row["path"])
                if target.exists():
                    target.unlink()
            self.conn.execute("DELETE FROM assets WHERE id = ?", (row["id"],))
            removed.append(row["path"])
        self.conn.commit()
        if removed:
            log("bank.prune", removed=len(removed), unused_days=unused_days)
        return removed
