"""Config em YAML, validada com Pydantic. Nada de valor hardcoded no codigo."""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from .schemas import SUB_SHOT_ORDER


class AnthropicConfig(BaseModel):
    model: str
    env: str
    effort: str = "high"
    max_tokens: int = 16000
    timeout_seconds: float = 900.0
    max_attempts: int = 3
    # Fase 1 do planejamento: uma chamada a mais que le a transcricao inteira
    # e devolve o briefing visual do video, antes de a fase 2 decidir cortes.
    two_phase: bool = True


class EditorialConfig(BaseModel):
    broll_min_seconds: float
    broll_max_seconds: float
    max_switches_per_minute: int
    intro_aroll_seconds: float
    broll_ratio_min: float
    broll_ratio_max: float
    concept_tags_min: int = 2
    concept_tags_max: int = 4


class ScriptConfig(BaseModel):
    """Estagio 1: da ideia para o roteiro.

    `words_per_minute` e o que converte roteiro em duracao ANTES de existir
    gravacao — e com isso o storyboard sabe quanto tempo de tela ele pode
    pedir. 150 e uma taxa de fala explicativa em portugues; meca a sua num
    video antigo (palavras do transcript / duracao em minutos) e ajuste, porque
    errar aqui faz o storyboard pedir cobertura para um video que nao existe.
    """

    words_per_minute: float = 150.0


class AlignConfig(BaseModel):
    """Estagio de alinhamento: onde cada beat visual caiu na fala real.

    `min_similarity` so entra em jogo quando a busca LITERAL falha — ou seja,
    quando voce parafraseou aquele trecho ao gravar. Abaixo dele o beat fica
    orfao e sai no relatorio, em vez de a imagem ser encaixada num lugar
    plausivel e errado. 0.55 e frouxo de proposito: um ancora de seis palavras
    contra um segmento de vinte tem cosseno naturalmente baixo.
    """

    min_similarity: float = 0.55


class TranscribeConfig(BaseModel):
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "pt"
    beam_size: int = 5
    vad_filter: bool = True


class TrimConfig(BaseModel):
    """Corte seco de pausa longa e hesitacao.

    A lista de `fillers` NAO e aplicada por texto puro. "e" e a 3a pessoa de
    "ser" e "um" e artigo: cortar por token destruiria frases inteiras. Todo
    candidato precisa passar tambem pelo silencio ANTES do som e pela duracao
    — ver o docstring de `trim.py` para por que o silencio e medido de um
    lado so, e o que a duracao e comparada contra.
    """

    enabled: bool = True
    pause_min_seconds: float = 0.8
    # Teto para TODA pausa, nao limiar de remocao. Zero desliga. Ver o
    # docstring de `trim.py`: e o unico lever que funciona em fala fluente,
    # onde nao ha trecho morto para remover.
    pause_max_seconds: float = 0.0
    filler_min_seconds: float = 0.40
    filler_silence_seconds: float = 0.25
    # Quantas vezes a mediana daquele token na propria fala. E o que separa
    # verbo abrindo frase de hesitacao alongada, sem depender de um numero
    # absoluto que muda com o ritmo de quem gravou.
    filler_stretch_ratio: float = 2.0
    # Deixado em cada ponta do corte, para a emenda nao soar cortada rente.
    keep_margin_seconds: float = 0.12
    # Trecho mantido menor que isso e absorvido em vez de virar fragmento.
    min_keep_seconds: float = 0.35
    fillers: list[str] = Field(
        default_factory=lambda: [
            "e", "eh", "ehh", "ah", "ahn", "ahm", "a",
            "hm", "hmm", "hum", "uhm", "um", "mm", "mmm", "uh",
        ]
    )


class BankConfig(BaseModel):
    db_path: str
    images_dir: str
    embedding_model: str
    similarity_threshold: float


class NetworkConfig(BaseModel):
    """Retries por tipo de chamada.

    A distincao nao e cosmetica. Criar uma predicao no provider NAO e
    idempotente: se ela roda no servidor e a resposta se perde, repetir gera
    uma segunda imagem e cobra duas vezes. Ja baixar um arquivo por URL e
    idempotente e de graca, entao pode insistir muito mais.
    """

    api_attempts: int = 3
    download_attempts: int = 6
    base_delay_seconds: float = 1.0


class BudgetConfig(BaseModel):
    max_usd_per_video: float
    brl_per_usd: float


class ReplicateConfig(BaseModel):
    """O provider de imagem.

    `input` vai inteiro para o campo `input` da predicao, e o pipeline so
    acrescenta o `prompt` (que e dele) e o `output_format` (que ele precisa
    conhecer para nomear o arquivo). Era um dict fixo no codigo com
    `num_outputs: 1` dentro, o que tornava "provider plugavel" falso: o
    `flux-1.1-pro`, por exemplo, nao aceita `num_outputs` e recusa a
    requisicao. Configuravel, trocar de modelo e uma linha de YAML — e os
    knobs que mudam a IMAGEM sem mudar o conceito (`guidance`,
    `num_inference_steps`, `megapixels`) ficam na sua mao sem passar por aqui.
    """

    env: str
    model: str
    cost_usd_per_image: float
    # Fica de fora do `input` porque quem usa e o pipeline: e a extensao do
    # arquivo que o banco e o render vao abrir.
    output_format: str = "png"
    input: dict[str, object] = Field(
        default_factory=lambda: {"aspect_ratio": "16:9", "num_outputs": 1}
    )
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
    # Usado so quando ha corte: sem corte o audio sai por stream copy.
    audio_bitrate_kbps: int = 192
    crossfade_seconds: float = 0.4
    # Fracao maxima do segmento que cada fade pode ocupar. Sem teto, um b-roll
    # de 0.6s tem pico de opacidade em 2/3 e zero frame na tela (medido). 0.25
    # garante metade do segmento em opacidade cheia qualquer que seja a
    # duracao. Ver `fade_for` em filtergraph.py para os numeros medidos.
    max_fade_ratio: float = 0.25
    solid_fallback_color: str = "0x1B2A33"
    # Medido: 24 planos (6 imagens com 4 sub-planos cada) dao 9092 chars, e
    # as emendas do aperto de pausa somam por cima disso. Ver o comentario no
    # config de exemplo para por que 16000 e nao 3000, e qual e o teto real.
    max_filtergraph_chars: int = 16000
    # --- sub-planos -------------------------------------------------------
    # Varios planos tirados da MESMA imagem, recortando regioes diferentes.
    # O teto de 4 e geometrico e nao de gosto: com `canvas_scale: 2` o
    # `prep_size` e `largura * 2 * zoom_max`, entao metade dele e exatamente
    # `largura * zoom_max` — o quadrante cabe nativo e um terco ampliaria 1.5x.
    # Nove sub-planos exigiriam `canvas_scale: 3`, ou seja uma imagem de
    # 6453px que nenhum provider entrega.
    max_sub_shots: int = 4
    # Zero desliga: cada faixa de b-roll vira um plano so, como era antes.
    sub_shot_seconds: float = 2.5
    # Avisa quando a imagem gerada e pequena demais para o plano em que ela
    # vai aparecer. Ver `upscale_factor` em stages/render.py: o mesmo arquivo
    # amplia o DOBRO num quadrante, e e esse numero que responde se o
    # "delirio" percebido e aderencia do modelo ou falta de resolucao.
    upscale_warn_factor: float = 1.5
    ken_burns: KenBurnsConfig = Field(default_factory=KenBurnsConfig)

    @model_validator(mode="after")
    def _sub_planos_cabem_na_geometria(self) -> RenderConfig:
        """Duas condicoes geometricas, checadas antes do primeiro ffmpeg.

        Nenhuma das duas quebraria o render: as duas produziriam video pior em
        silencio, depois de o dinheiro das imagens ja ter sido gasto.

        `max_sub_shots` acima do numero de regioes que existem faria dois
        planos recortarem o mesmo pedaco da imagem. O corte seco entre eles
        nao pareceria um corte — pareceria o movimento saltando dentro da
        mesma moldura, que e pior que nao cortar.

        `canvas_scale` abaixo de 2 nao tem meio para dar ao quadrante: o
        quadrante sai com metade da tela preparada, e para preencher a saida
        ele teria que ser ampliado. E exatamente o que `prep_size` existe para
        evitar, e a invariante e `canvas_scale: 2` <=> quadrante nativo.
        """
        if self.sub_shot_seconds <= 0 or self.max_sub_shots <= 1:
            return self
        if self.max_sub_shots > len(SUB_SHOT_ORDER):
            raise ValueError(
                f"render.max_sub_shots: {self.max_sub_shots} planos, mas so existem "
                f"{len(SUB_SHOT_ORDER)} regioes distintas na imagem "
                f"({', '.join(SUB_SHOT_ORDER)}) — dois planos recortariam o mesmo pedaco"
            )
        if self.ken_burns.canvas_scale < 2:
            raise ValueError(
                f"render.ken_burns.canvas_scale: {self.ken_burns.canvas_scale} nao "
                "sustenta sub-plano — o quadrante sairia com metade da tela de "
                "trabalho e precisaria ser ampliado para preencher a saida. "
                "Use canvas_scale: 2, ou render.sub_shot_seconds: 0 para desligar"
            )
        return self


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
    # Um dos dois: `style_suffix` literal, ou `style` nomeando uma entrada de
    # `styles`. Depois do validador, `style_suffix` sempre carrega o texto
    # resolvido, entao o resto do pipeline nao sabe qual caminho foi usado —
    # inclusive a chave de cache dos assets, que ja dependia dele.
    style_suffix: str = ""
    style: str = ""
    styles: dict[str, str] = Field(default_factory=dict)
    anthropic: AnthropicConfig
    editorial: EditorialConfig
    script: ScriptConfig = Field(default_factory=ScriptConfig)
    align: AlignConfig = Field(default_factory=AlignConfig)
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    trim: TrimConfig = Field(default_factory=TrimConfig)
    bank: BankConfig
    budget: BudgetConfig
    image_provider: ImageProviderConfig
    stock: StockConfig
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    subtitles: SubtitlesConfig = Field(default_factory=SubtitlesConfig)

    # Preenchido no load, nao vem do YAML: tudo que e caminho relativo no
    # config resolve contra a raiz do projeto, nao contra o cwd.
    root: Path = Field(default=Path("."), exclude=True)

    @model_validator(mode="after")
    def _resolve_style(self) -> Config:
        """Resolve `style` + `styles` para o texto final de `style_suffix`.

        Falha alto e cedo de proposito: estilo errado nao quebra nada, so
        produz um video com as imagens erradas — e ai o dinheiro da geracao
        ja foi gasto. Um nome digitado errado tem que parar o pipeline antes
        da primeira chamada, nao virar string vazia.
        """
        if self.style and self.style_suffix:
            raise ValueError(
                "defina `style` (nome dentro de `styles`) OU `style_suffix` "
                "(o texto literal), nao os dois"
            )
        if self.style:
            if self.style not in self.styles:
                disponiveis = ", ".join(sorted(self.styles)) or "(nenhum)"
                raise ValueError(
                    f"style: '{self.style}' nao existe em `styles`. "
                    f"Disponiveis: {disponiveis}"
                )
            self.style_suffix = self.styles[self.style]
        if not self.style_suffix.strip():
            raise ValueError(
                "nenhum estilo definido: preencha `style` (com `styles`) ou `style_suffix`"
            )
        return self

    @property
    def style_name(self) -> str:
        """Identidade do estilo, para o banco nao reusar imagem de outro.

        Com `styles` e o nome; com `style_suffix` literal e um hash do texto,
        porque o texto inteiro seria uma chave grande demais para guardar em
        cada linha e comparar. O que importa e ser estavel e distinguir.
        """
        if self.style:
            return self.style
        return "suffix:" + sha256(self.style_suffix.strip().encode()).hexdigest()[:12]

    @property
    def known_styles(self) -> dict[str, str]:
        """Nome -> sufixo de todo estilo que este config conhece.

        Com `styles` e o proprio mapa; com `style_suffix` literal e a unica
        entrada. O banco usa isto na migracao para reconhecer a qual estilo
        cada imagem antiga pertence, pelo `prompt` que ficou guardado.
        """
        return dict(self.styles) if self.styles else {self.style_name: self.style_suffix}

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
