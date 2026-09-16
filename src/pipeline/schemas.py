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

    @property
    def duration(self) -> float:
        return self.end - self.start


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
# 2b. trim -> trim.json
#
# O corte muda a linha de tempo, entao este estagio tambem devolve o
# transcript remapeado. Tudo a jusante trabalha na timeline nova e nao precisa
# saber que houve corte.
# --------------------------------------------------------------------------


class Cut(BaseModel):
    start: float                 # tempo na ENTRADA original
    end: float
    reason: Literal["pause", "filler", "squeeze"]
    token: str = ""              # so para filler: a palavra removida
    context: str = ""            # texto ao redor, para voce revisar o corte

    @property
    def duration(self) -> float:
        return self.end - self.start


class KeepRange(BaseModel):
    start: float                 # tempo na ENTRADA original
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class TrimStats(BaseModel):
    original_seconds: float
    trimmed_seconds: float
    removed_seconds: float
    removed_ratio: float
    n_cuts: int
    n_pause_cuts: int
    n_filler_cuts: int
    n_squeeze_cuts: int = 0


class TrimPlan(BaseModel):
    input_hash: str              # digest do transcript + regras de corte
    enabled: bool
    keep: list[KeepRange]
    cuts: list[Cut]
    stats: TrimStats

    def map_time(self, original: float) -> float:
        """Converte um instante da entrada para a timeline cortada."""
        elapsed = 0.0
        for span in self.keep:
            if original < span.start:
                return elapsed
            if original <= span.end:
                return elapsed + (original - span.start)
            elapsed += span.duration
        return elapsed

    def original_time(self, trimmed: float) -> float:
        """O inverso: um instante da saida de volta para a entrada.

        O render precisa dos dois sentidos. Os trechos a manter vivem no
        tempo da ENTRADA; a EDL, os overlays e os chunks vivem no tempo
        CORTADO. Sem o inverso nao ha como saber que janela da entrada um
        chunk precisa ler.
        """
        elapsed = 0.0
        for span in self.keep:
            if trimmed <= elapsed + span.duration:
                return span.start + (trimmed - elapsed)
            elapsed += span.duration
        return self.keep[-1].end if self.keep else trimmed

    def keep_within(self, first: float, last: float) -> list[tuple[float, float]]:
        """Trechos a manter que um chunk de [first, last) da saida precisa.

        Devolvidos em tempo local da janela de entrada que o chunk vai abrir,
        que e o que as expressoes de `trim` esperam depois do `-ss`.
        """
        window_start = self.original_time(first)
        window_end = self.original_time(last)
        local: list[tuple[float, float]] = []
        for span in self.keep:
            lo, hi = max(span.start, window_start), min(span.end, window_end)
            if hi > lo:
                local.append((lo - window_start, hi - window_start))
        return local


# --------------------------------------------------------------------------
# 3. plan -> edl.json
#
# O modelo devolve faixas de INDICE de segmento do transcript, nao tempos.
# Assim "nunca cortar no meio de uma frase" deixa de ser uma regra que da
# para violar: as fronteiras sao, por construcao, fronteiras de segmento.
# --------------------------------------------------------------------------


class Brief(BaseModel):
    """Fase 1 do planejamento: o que o video e, antes de decidir cortes.

    Existe porque as duas tarefas competiam na mesma resposta. Pedir na mesma
    chamada "segmente a timeline" e "invente as imagens" fazia a segunda
    sofrer: saiam cenas plausiveis para um video de criador genericamente, nao
    para ESTE video.

    O briefing define o vocabulario visual; cada `concept` continua ancorado
    ao proprio trecho. Uma imagem que ilustra o tema geral mas nao o que esta
    sendo dito naquele momento e pior que uma imagem generica — o espectador
    sente o descolamento.
    """

    subject: str = Field(description="do que o video trata, em uma frase")
    argument: str = Field(description="o argumento central que ele defende, em uma frase")
    audience_takeaway: str = Field(description="o que o espectador leva embora")
    visual_vocabulary: list[str] = Field(
        description="6 a 10 elementos visuais concretos em ingles (objetos, "
                    "ambientes, materiais, tipo de luz) que atravessam o video "
                    "inteiro e dao unidade as imagens"
    )
    avoid: list[str] = Field(
        description="2 a 5 clices visuais em ingles a evitar neste video "
                    "especifico, por serem obvios ou nao dizerem nada"
    )


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
    brief: Brief | None = None
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


# --------------------------------------------------------------------------
# Roteiro primeiro: script.md -> storyboard.json -> edl.json
#
# Inverte a ordem do pipeline antigo. Antes a intencao visual era INFERIDA de
# fala improvisada; aqui ela e escrita e aprovada antes de a camera ligar.
# Ver docs/plans/2026-09-16-roteiro-primeiro.md.
# --------------------------------------------------------------------------


class ScriptBeat(BaseModel):
    """Um trecho do roteiro. O `id` vem do cabecalho, nao da posicao."""

    id: int
    title: str = ""
    text: str


class Script(BaseModel):
    slug: str
    subject: str
    argument: str
    viewer_takeaway: str
    duration_target_seconds: float = 0.0
    beats: list[ScriptBeat]

    def beat(self, beat_id: int) -> ScriptBeat | None:
        return next((b for b in self.beats if b.id == beat_id), None)

    def digest(self) -> str:
        """Identidade do roteiro, para a chave de cache do storyboard.

        Inclui o TEXTO dos beats, nao so os ids: editar o texto de um beat
        muda o que a imagem dele deveria mostrar, entao tem que invalidar.
        """
        from .util import text_hash

        return text_hash(
            self.subject, self.argument,
            *(f"{b.id}:{b.text}" for b in self.beats),
        )


class PlannedBeat(BaseModel):
    """Um beat visual, como o modelo devolve. Sem tempo absoluto nenhum.

    `script_anchor` e a peca que faz o desenho sobreviver a improviso: e um
    trecho do PROPRIO roteiro, e o estagio de alinhamento descobre onde ele
    caiu na fala real. Pedir segundo absoluto aqui seria pedir ao modelo para
    prever a sua entrega, e qualquer desvio deslocaria tudo adiante.
    """

    beat_id: int = Field(description="o id do beat do roteiro que esta imagem ilustra")
    script_anchor: str = Field(
        description="um trecho curto e literal do texto daquele beat, copiado "
                    "exatamente como esta escrito, que marca onde a imagem entra"
    )
    concept: str = Field(
        description="descricao visual concreta em ingles do que a imagem mostra. "
                    "Sem texto na imagem, sem rosto em foco, sem metafora abstrata"
    )
    concept_tags: list[str] = Field(
        description="2 a 4 palavras-chave em ingles que resumem a cena"
    )
    sub_shots: int = Field(
        description="quantos planos tirar desta mesma imagem, de 1 a 4, "
                    "recortando regioes diferentes dela"
    )
    rationale: str = Field(
        description="por que esta imagem, neste ponto, em uma frase — e o que "
                    "voce le na revisao para decidir se aprova"
    )


class PlannedStoryboard(BaseModel):
    beats: list[PlannedBeat]


class StoryboardBeat(PlannedBeat):
    """O beat validado, com o que o pipeline derivou dele."""

    anchor_offset: int = Field(
        default=-1,
        description="posicao do ancora no texto do beat; -1 quando nao foi "
                    "encontrado literalmente e caiu na busca por similaridade",
    )
    seconds_per_shot: float = 0.0

    @property
    def screen_seconds(self) -> float:
        return self.sub_shots * self.seconds_per_shot


class Storyboard(BaseModel):
    input_hash: str          # digest do roteiro + regras + estilo
    beats: list[StoryboardBeat]

    @property
    def total_screen_seconds(self) -> float:
        return sum(b.screen_seconds for b in self.beats)

    @property
    def n_images(self) -> int:
        """Uma imagem por beat — os sub-planos saem todos da mesma."""
        return len(self.beats)


Region = Literal["full", "top_left", "top_right", "bottom_left", "bottom_right"]

# A ordem em que os sub-planos varrem a imagem, e ao mesmo tempo o teto de
# quantos existem. Comeca no plano cheio para o espectador ver o conjunto
# antes do detalhe, e os quadrantes seguem em diagonal para que dois planos
# consecutivos nunca compartilhem uma borda — com `top_left` seguido de
# `top_right` o corte seco pareceria um pan aos saltos em vez de um corte.
SUB_SHOT_ORDER: tuple[Region, ...] = (
    "full", "top_left", "bottom_right", "top_right", "bottom_left",
)


class SubShot(BaseModel):
    """Um plano tirado de uma regiao da imagem, ja na timeline.

    A regiao e escolhida pelo pipeline e nao pelo modelo: o limite e
    geometrico. Ver `canvas_for` e `region_chain` em `filtergraph.py` — com
    `canvas_scale: 2` o quadrante e exatamente nativo em 1080p, e um terco
    ampliaria 1.5x.
    """

    index: int
    start: float
    end: float
    region: Region
    direction: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class PlannedScript(BaseModel):
    """O roteiro como o modelo devolve. Sem `slug`: quem nomeia e o pipeline.

    O modelo devolve estrutura e nao Markdown pronto de proposito — assim a
    validacao e de graca, e o `render` do `script.py` e o unico lugar que sabe
    escrever o formato.
    """

    subject: str = Field(description="do que o video trata, em uma frase")
    argument: str = Field(description="o argumento central que ele defende, em uma frase")
    viewer_takeaway: str = Field(description="o que o espectador leva embora")
    beats: list[ScriptBeat] = Field(
        description="os trechos do roteiro, com id comecando em 1 e sem buraco"
    )


class BeatPlacement(BaseModel):
    """Onde um beat visual caiu na fala real.

    `method` fica gravado porque muda como voce le o resultado: `literal`
    quer dizer que voce disse aquele trecho como estava escrito; `semantic`
    que voce parafraseou e o embedding achou o lugar; `unaligned` que a imagem
    ficou orfa — nao existe um lugar bom para ela, e forcar um seria pior.
    """

    beat_id: int
    anchor_offset: int
    segment: int             # indice no transcript; -1 quando orfao
    method: Literal["literal", "semantic", "unaligned"]
    similarity: float
