"""Schemas dos artefatos que trafegam entre estagios.

Cada artefato carrega `input_hash`: o hash daquilo de que ele foi derivado.
E so isso que faz o pipeline ser resumivel — um estagio le o proprio artefato,
compara o hash e decide se tem trabalho a fazer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# 1. ingest -> manifest.json
# --------------------------------------------------------------------------


class Manifest(BaseModel):
    input_hash: str          # hash do arquivo de entrada
    slug: str
    source_path: str
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str
    audio_path: str          # WAV 16kHz mono, relativo a raiz do projeto
    audio_hash: str
    created_at: str


# --------------------------------------------------------------------------
# 2. transcribe -> transcript.json
# --------------------------------------------------------------------------


class Word(BaseModel):
    start: float
    end: float
    word: str


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    input_hash: str          # hash do audio
    language: str
    model: str
    duration: float
    segments: list[TranscriptSegment]

    def digest(self) -> str:
        """Identidade do transcript, para o estagio 3 cachear em cima dele."""
        from .util import text_hash

        return text_hash(self.input_hash, self.language, len(self.segments))


# --------------------------------------------------------------------------
# 3. plan -> edl.json
#
# O modelo devolve faixas de INDICE de segmento do transcript, nao tempos.
# Assim "nunca cortar no meio de uma frase" deixa de ser uma regra que da
# para violar: as fronteiras sao, por construcao, fronteiras de segmento.
# --------------------------------------------------------------------------


class PlannedSpan(BaseModel):
    """Uma faixa como o modelo a devolve.

    Nenhum campo tem default: structured outputs exige todo campo em
    `required`, entao o modelo preenche `concept=""` e `concept_tags=[]` nas
    faixas de a-roll — e o validador cobra justamente isso delas.
    """

    kind: Literal["aroll", "broll"] = Field(
        description="aroll mostra a pessoa falando; broll mostra imagem ilustrativa"
    )
    first_segment: int = Field(description="indice do primeiro segmento do transcript nesta faixa")
    last_segment: int = Field(description="indice do ultimo segmento do transcript nesta faixa, inclusive")
    concept: str = Field(
        description="apenas para broll: descricao visual em ingles, concreta, "
                    "sem texto na imagem. String vazia para aroll."
    )
    concept_tags: list[str] = Field(
        description="apenas para broll: 2 a 4 palavras-chave em ingles para busca "
                    "em banco de stock. Lista vazia para aroll."
    )


class PlannedEDL(BaseModel):
    """O formato exato que a API da Anthropic preenche."""

    spans: list[PlannedSpan]


class EDLSegment(PlannedSpan):
    """Uma faixa depois de resolvida contra o transcript."""

    start: float
    end: float

    # dentro do pipeline estes campos sao opcionais de novo
    concept: str = ""
    concept_tags: list[str] = Field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


class EDLStats(BaseModel):
    broll_ratio: float
    switches_per_minute: float       # taxa media no video inteiro
    switches_per_minute_max: float   # pior janela deslizante de 60s
    n_aroll: int
    n_broll: int
    broll_seconds: float


class EDL(BaseModel):
    input_hash: str          # digest do transcript
    model: str
    attempts: int
    duration: float
    segments: list[EDLSegment]
    stats: EDLStats

    @property
    def broll(self) -> list[EDLSegment]:
        return [s for s in self.segments if s.kind == "broll"]

    def digest(self) -> str:
        from .util import text_hash

        return text_hash(
            self.input_hash,
            *(f"{s.kind}:{s.start:.3f}:{s.end:.3f}:{s.concept}" for s in self.segments),
        )


# --------------------------------------------------------------------------
# 4. assets -> assets.json
# --------------------------------------------------------------------------

Origin = Literal["bank", "stock", "generated", "solid"]


class AssetItem(BaseModel):
    segment_index: int       # indice em EDL.segments
    origin: Origin
    path: str | None = None  # None apenas para origin="solid"
    asset_id: int | None = None
    prompt: str | None = None
    similarity: float | None = None
    cost_usd: float = 0.0
    note: str | None = None


class AssetEstimate(BaseModel):
    n_broll: int
    n_from_bank: int
    n_from_stock: int
    n_to_generate: int
    worst_case_usd: float    # tudo gerado
    estimated_usd: float
    budget_usd: float
    within_budget: bool


class Assets(BaseModel):
    input_hash: str          # digest da EDL
    dry_run: bool = False
    items: list[AssetItem]
    estimate: AssetEstimate
    total_cost_usd: float = 0.0
    by_origin: dict[str, int] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# 5. render -> final.mp4 (+ filtergraph.txt, subs.ass)
# 6. report -> report.json
# --------------------------------------------------------------------------


class Report(BaseModel):
    input_hash: str
    slug: str
    output_path: str
    duration_seconds: float
    n_aroll: int
    n_broll: int
    broll_ratio: float
    reuse_rate: float        # fracao dos b-rolls resolvida sem custo
    by_origin: dict[str, int]
    cost_usd: float
    cost_brl: float
    brl_per_usd: float
    cost_usd_per_final_minute: float
    generated_at: str
