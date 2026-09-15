"""Config em YAML, validada com Pydantic. Nada de valor hardcoded no codigo."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AnthropicConfig(BaseModel):
    model: str
    env: str
    effort: str = "high"
    max_tokens: int = 16000
    timeout_seconds: float = 900.0
    max_attempts: int = 3


class EditorialConfig(BaseModel):
    broll_min_seconds: float
    broll_max_seconds: float
    max_switches_per_minute: int
    intro_aroll_seconds: float
    broll_ratio_min: float
    broll_ratio_max: float
    concept_tags_min: int = 2
    concept_tags_max: int = 4


class TranscribeConfig(BaseModel):
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "pt"
    beam_size: int = 5
    vad_filter: bool = True


class BankConfig(BaseModel):
    db_path: str
    images_dir: str
    embedding_model: str
    similarity_threshold: float


class BudgetConfig(BaseModel):
    max_usd_per_video: float
    brl_per_usd: float


class ReplicateConfig(BaseModel):
    env: str
    model: str
    cost_usd_per_image: float
    aspect_ratio: str = "16:9"
    output_format: str = "png"
    timeout_seconds: float = 120.0


class ImageProviderConfig(BaseModel):
    active: str
    replicate: ReplicateConfig


class StockConfig(BaseModel):
    provider: str = "pexels"
    env: str
    orientation: str = "landscape"
    min_width: int = 2400
    timeout_seconds: float = 30.0
    generic_tags: list[str] = Field(default_factory=list)


class KenBurnsConfig(BaseModel):
    engine: str = "scale_crop"
    zoom_min: float = 1.0
    zoom_max: float = 1.12
    directions: list[str] = Field(default_factory=lambda: ["zoom_in", "pan_right", "zoom_out", "pan_left"])
    canvas_scale: int = 2


class RenderConfig(BaseModel):
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    crf: int = 20
    preset: str = "medium"
    crossfade_seconds: float = 0.4
    solid_fallback_color: str = "0x1B2A33"
    max_filtergraph_chars: int = 3000
    ken_burns: KenBurnsConfig = Field(default_factory=KenBurnsConfig)


class SubtitlesConfig(BaseModel):
    enabled: bool = True
    font_name: str = "Inter"
    font_size: int = 54
    primary_color: str = "&H00FFFFFF"
    outline_color: str = "&H00101010"
    outline: int = 3
    shadow: int = 1
    bold: bool = True
    alignment: int = 2
    margin_v: int = 90
    max_chars_per_line: int = 42
    max_lines: int = 2


class Config(BaseModel):
    style_suffix: str
    anthropic: AnthropicConfig
    editorial: EditorialConfig
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    bank: BankConfig
    budget: BudgetConfig
    image_provider: ImageProviderConfig
    stock: StockConfig
    render: RenderConfig = Field(default_factory=RenderConfig)
    subtitles: SubtitlesConfig = Field(default_factory=SubtitlesConfig)

    # Preenchido no load, nao vem do YAML: tudo que e caminho relativo no
    # config resolve contra a raiz do projeto, nao contra o cwd.
    root: Path = Field(default=Path("."), exclude=True)

    def path(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else self.root / candidate

    @property
    def work_dir(self) -> Path:
        return self.root / "work"


def load_config(path: Path | str) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = Config.model_validate(raw)
    config.root = path.parent
    return config


def require_key(env_var: str, what: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise RuntimeError(f"{what}: variavel de ambiente {env_var} nao esta definida")
    return key
