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
    embedding_model TEXT  NOT NULL DEFAULT '',
    style         TEXT    NOT NULL DEFAULT '',
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


# Assinaturas de TLS corrompido. Sao o caso em que "tente de novo" e
# conselho ruim: o problema nao e transitorio, e algo no meio do caminho
# quebrando os registros TLS — antivirus, VPN ou proxy inspecionando a
# conexao. Arquivos pequenos passam; transferencia sustentada quebra.
_TLS_MARKERS = (
    "DECRYPTION_FAILED_OR_BAD_RECORD_MAC",
    "BAD_RECORD_MAC",
    "SSLError",
    "record layer failure",
)


def _embedder_remedies(exc: Exception) -> str:
    """Orientacao especifica para o modo de falha observado.

    "Tente de novo" e conselho ruim para TLS corrompido: o problema nao e
    transitorio. Arquivo pequeno passa, transferencia sustentada quebra.
    """
    text = f"{type(exc).__name__}: {exc}"

    if any(marker in text for marker in _TLS_MARKERS):
        return """\
O erro e de TLS corrompido, nao falta de rede: algo entre voce e o
HuggingFace esta quebrando os registros da conexao. Reexecutar nao
resolve, porque arquivo pequeno passa e transferencia sustentada quebra.

Em ordem de eficacia:

  1. Baixe com o downloader em Rust, que usa outra pilha TLS:
       uv pip install hf_transfer
       set HF_HUB_ENABLE_HF_TRANSFER=1

  2. Desligue a inspecao de HTTPS do antivirus, ou a VPN, so durante o
     download. E a causa mais comum deste erro.

  3. Baixe fora do pipeline, repetindo ate completar (cada tentativa
     retoma de onde parou):
       hf download <modelo>

  4. Ultimo recurso: baixe o repositorio do modelo pelo navegador e
     aponte `bank.embedding_model` para a pasta local.
"""

    return """\
E download do HuggingFace, entao costuma ser transitorio: rodar de novo
retoma de onde parou. Se insistir:

  set HF_HUB_DISABLE_XET=1        (troca para o download classico)
  uv pip install hf_transfer && set HF_HUB_ENABLE_HF_TRANSFER=1

`bank.embedding_model` tambem aceita o caminho de uma pasta local, se
voce preferir baixar o modelo por fora.
"""


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
        self, root: Path, db_path: Path, images_dir: Path, embedder: Embedder,
        threshold: float, style: str = "", styles: dict[str, str] | None = None,
    ) -> None:
        # Os caminhos em `assets.path` sao guardados relativos a `root`, para o
        # banco continuar valido se o projeto mudar de lugar.
        self.root = Path(root)
        self.db_path = Path(db_path)
        self.images_dir = Path(images_dir)
        self.embedder = embedder
        self.threshold = threshold
        # Identidade do estilo ativo. O reuso e limitado a ele: ver find_similar.
        self.style = style
        # Nome -> sufixo de todos os estilos conhecidos, usado SO pela
        # migracao, para reconhecer no `prompt` guardado a qual estilo cada
        # linha antiga pertence. O mapa inteiro e nao so o ativo: quem mantem
        # dois estilos no config e troca entre eles nao perde o reuso do que
        # nao esta ativo agora.
        self.styles = dict(styles or {})
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    @property
    def model_name(self) -> str:
        """Identifica o espaco vetorial das comparacoes."""
        return getattr(self.embedder, "model_name", "unknown")

    def _migrate(self) -> None:
        """Banco criado antes da coluna `embedding_model`.

        As linhas existentes recebem o modelo atual: e o que quase certamente
        as produziu, e o alternativo — deixar em branco — orfanaria todo o
        banco de quem ja tinha assets.
        """
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(assets)")}
        if "embedding_model" not in columns:
            self.conn.execute(
                "ALTER TABLE assets ADD COLUMN embedding_model TEXT NOT NULL DEFAULT ''"
            )
        if "style" not in columns:
            self.conn.execute(
                "ALTER TABLE assets ADD COLUMN style TEXT NOT NULL DEFAULT ''"
            )
            # Aqui NAO se assume o estilo atual, ao contrario do modelo acima,
            # e a diferenca importa. Para `embedding_model` assumir o valor
            # atual e seguro: uma maquina usou um modelo. Para `style` nao:
            # quem roda esta migracao provavelmente esta estreando um estilo
            # novo, e marcar as linhas antigas com ele faria o banco jurar que
            # uma imagem warm-sand e old money — e reusa-la, que e o dano
            # exato que a coluna existe para evitar.
            #
            # Mas orfanar tudo tambem e caro: o reuso e requisito de custo, nao
            # otimizacao. Da para ser exato em vez de chutar — `prompt` guarda
            # `concept + style_suffix`, entao a linha cujo prompt termina com o
            # estilo ativo E daquele estilo, lido do que de fato foi enviado ao
            # provider. As outras viram '?', que nao casa com estilo nenhum.
            #
            # Stock fica em '' e continua valendo para qualquer estilo: e foto.
            recovered = 0
            for name, suffix in self.styles.items():
                if not suffix.strip():
                    continue
                recovered += self.conn.execute(
                    "UPDATE assets SET style = ? WHERE origin = 'generated'"
                    " AND style = '' AND prompt IS NOT NULL"
                    " AND TRIM(prompt) LIKE '%' || ?",
                    (name, suffix.strip()),
                ).rowcount
            orphaned = self.conn.execute(
                "UPDATE assets SET style = '?' WHERE origin = 'generated' AND style = ''"
            ).rowcount
            if recovered or orphaned:
                log("bank.migrated", reconhecidas=recovered, orfas=orphaned,
                    detail="orfa e imagem gerada cujo prompt nao casa com o estilo "
                           "ativo; nao sera reusada (limpe com `pipeline bank prune`)")
        pending = self.conn.execute(
            "SELECT COUNT(*) AS n FROM assets WHERE embedding_model = ''"
        ).fetchone()["n"]
        if pending:
            self.conn.execute(
                "UPDATE assets SET embedding_model = ? WHERE embedding_model = ''",
                (self.model_name,),
            )
            log("bank.migrated", rows=pending, assumed_model=self.model_name)

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

    def warmup(self) -> None:
        """Carrega o modelo de embedding ANTES de qualquer gasto.

        O embedder e lazy e so era tocado no primeiro `add`, que acontece
        depois de baixar ou gerar a imagem. Com o banco vazio o
        curto-circuito de `find_similar` adiava ainda mais. Resultado: um
        download de 400MB que falha (rede, disco, rate limit do HF) derrubava
        o estagio depois de a imagem ja ter sido paga.

        Chamado no inicio do estagio 4 numa execucao real, o erro aparece
        antes de gastar.
        """
        try:
            self.embedder.embed("warmup")
        except Exception as exc:
            raise RuntimeError(
                f"nao foi possivel carregar o modelo de embedding "
                f"'{self.model_name}' ({type(exc).__name__}: {exc}).\n"
                f"\nNada foi gasto.\n" + _embedder_remedies(exc)
            ) from exc

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
        if not self.conn.execute(
            "SELECT 1 FROM assets WHERE embedding_model = ? AND (style = ? OR origin = 'stock')"
            " LIMIT 1", (self.model_name, self.style),
        ).fetchone():
            return None

        target = self.embedder.embed(concept)
        best: tuple[Asset, float] | None = None

        # Compara so contra vetores do MESMO modelo. Dois modelos diferentes
        # podem ter a mesma dimensao (all-MiniLM-L6-v2 e o multilingue tem
        # 384 os dois), entao a checagem de tamanho no cosseno nao pega a
        # troca: seriam similaridades sem sentido, reusando imagem errada em
        # silencio.
        # Filtra tambem por ESTILO. `style_suffix` existe para 20 imagens
        # geradas em 20 chamadas parecerem do mesmo ilustrador; reusar uma
        # imagem de outro estilo desfaz exatamente isso, e em silencio — o
        # embedding e do `concept`, que nao carrega estilo nenhum. Mesma
        # familia do filtro por `embedding_model` acima.
        #
        # Stock entra em qualquer estilo: e foto, nao tem estilo de ilustracao
        # para casar ou destoar.
        rows = self.conn.execute(
            "SELECT * FROM assets WHERE embedding_model = ? AND (style = ? OR origin = 'stock')",
            (self.model_name, self.style),
        )
        for row in rows:
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
                                   embedding_model, style, cost_usd, created_at,
                                   use_count, last_used_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
            # Stock nao carrega estilo: e foto, e vale para qualquer um.
            (path, concept, prompt, origin, pack(vector), len(vector),
             self.model_name, "" if origin == "stock" else self.style,
             cost_usd, now_iso()),
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
        by_model = {
            r["embedding_model"]: r["n"]
            for r in self.conn.execute(
                "SELECT embedding_model, COUNT(*) AS n FROM assets GROUP BY embedding_model")
        }
        by_style = {
            r["style"]: r["n"]
            for r in self.conn.execute(
                "SELECT style, COUNT(*) AS n FROM assets WHERE origin = 'generated'"
                " GROUP BY style")
        }
        return {
            "assets": row["n"],
            "by_origin": by_origin,
            "by_model": by_model,
            "active_model": self.model_name,
            "by_style": by_style,
            "active_style": self.style,
            "total_spent_usd": round(row["spent"], 4),
            "total_uses": row["uses"],
            "never_used": never,
            "avg_uses_per_asset": round(row["uses"] / row["n"], 2) if row["n"] else 0.0,
            "db_path": str(self.db_path),
        }

    def forget(self, asset_id: int, *, delete_file: bool = True) -> bool:
        """Apaga UM asset do banco, e o arquivo dele.

        Existe para a regeracao. Uma imagem que voce rejeitou tem que sair do
        banco, nao so do video: deixar a linha la faz o proximo video com um
        conceito parecido reusar exatamente a imagem que voce recusou — e em
        silencio, porque reuso do banco nao passa por aprovacao nenhuma.
        """
        row = self.conn.execute(
            "SELECT path FROM assets WHERE id = ?", (asset_id,)).fetchone()
        if row is None:
            return False
        if delete_file:
            target = self.absolute(row["path"])
            if target.exists():
                target.unlink()
        self.conn.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
        self.conn.commit()
        log("bank.forget", asset=asset_id, path=row["path"])
        return True

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
