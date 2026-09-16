"""Construcao programatica do filtergraph.

Duas decisoes moldam este modulo:

**O audio nunca e tocado.** A trilha de video e a unica coisa que passa por
filtro; o audio sai do arquivo original por stream copy. Isso exige que a
duracao da saida seja exatamente a da entrada — e por isso que a transicao
NAO usa `xfade`, que encurta o video em 400ms por transicao e faria o audio
derivar. Em vez disso o video original e a camada base, rodando inteiro com
o timing intacto, e cada imagem de b-roll e sobreposta por cima com fade de
alpha na entrada e na saida. O fade das duas pontas e o crossfade nas duas
direcoes, e a duracao nao muda.

**Ken Burns sem `zoompan`.** O `zoompan` calcula o recorte em pixels
inteiros, entao num movimento lento a janela anda 1px a cada N frames em vez
de uma fracao de pixel por frame — e esse degrau e o tremor. Aqui o
movimento sai de dois filtros que o ffmpeg reavalia por frame: `scale` com
`eval=frame` para o zoom e as expressoes x/y do `crop` para a translacao.
Ambos operam numa tela de 2x a resolucao de saida e o downscale para 1080p
vem por ultimo, entao o passo de 1px vira meio pixel na saida.

Medido com ffmpeg 6.1 num pan de 1.12 sobre 2s: 58 de 58 frames mudaram, com
diferenca regular entre frames (coeficiente de variacao 0.05). Nao ha aqui
uma medicao do `zoompan` tremendo — o caso que o provoca e mais lento que o
que foi medido; o que sustenta a escolha e o mecanismo, nao um numero
comparativo.

Uma consequencia do `eval=frame`: o link de saida do `scale` muda de tamanho
a cada frame e o `crop` seguinte reconfigura. Verificado funcionando no
ffmpeg 6.1. Se a sua build reclamar, `render.ken_burns.engine: zoompan` no
config troca o motor.

**Sub-planos.** Uma faixa longa de b-roll nao precisa ser um plano so: o
mesmo arquivo pode dar varios planos, cada um recortando uma regiao
diferente da imagem. O recorte acontece antes do Ken Burns, na tela
preparada, e nao custa imagem nova — e por isso que ele existe, porque
imagem gerada e o unico item caro do pipeline.

A geometria fecha exatamente e nao por sorte: `prep_size` e
`largura * canvas_scale * zoom_max`, entao com `canvas_scale: 2` metade dele
e `largura * zoom_max` — o tamanho que o Ken Burns pede para nunca ampliar.
A invariante e `canvas_scale: 2` <=> quadrante nativo, qualquer que seja o
`zoom_max`. O que o quadrante perde e a margem de subpixel: no plano cheio o
passo de 1px do crop cai numa tela 2x e vira meio pixel na saida, no
quadrante a tela de trabalho JA e a saida e o passo e de 1px inteiro. Nao
aparece porque sub-plano e curto por construcao (`sub_shot_seconds`): a
2.5s o pan anda 3px por frame. Um sub-plano de 10s andaria 0.8px por frame e
ai o degrau apareceria.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

from .config import RenderConfig
from .log import log
from .schemas import SUB_SHOT_ORDER, Region


@dataclass
class Overlay:
    """Um b-roll posicionado na timeline do chunk (tempos locais)."""

    start: float
    end: float
    direction: str
    image_path: Path | None = None   # None -> fallback de cor solida
    region: Region = "full"
    # Fades de alpha das duas pontas, em segundos. `None` deriva da duracao
    # do proprio plano, que e o caso de um b-roll de um plano so. Entre
    # sub-planos da mesma imagem eles sao 0.0: ali o corte e seco, e um fade
    # faria a sua imagem reaparecer por 400ms no meio da faixa.
    fade_in: float | None = None
    fade_out: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Chunk:
    """Uma janela de tempo do video, renderizada numa chamada de ffmpeg."""

    index: int
    start: float                     # tempo global
    end: float                       # tempo global
    overlays: list[Overlay] = field(default_factory=list)
    # Trechos da ENTRADA a manter, em tempo local da janela. Vazio = sem corte.
    keep: list[tuple[float, float]] = field(default_factory=list)
    # A janela da ENTRADA que este chunk abre. Com corte, ela e maior que
    # [start, end), porque aqueles sao tempos da saida.
    input_start: float | None = None
    input_duration: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def frame_align(seconds: float, fps: float) -> float:
    """Encaixa o tempo no grid de frames. Sem isso as duracoes dos chunks nao
    somam exatamente a duracao original e o audio deriva na concatenacao."""
    if fps <= 0:
        return seconds
    return round(seconds * fps) / fps


# --------------------------------------------------------------------------
# Ken Burns
# --------------------------------------------------------------------------


def _ramp(start: float, end: float, duration: float) -> str:
    """Expressao ffmpeg de uma rampa linear de `start` a `end` sobre t."""
    if abs(end - start) < 1e-9:
        return f"{start:.6f}"
    return f"({start:.6f}+({end - start:.6f})*min(t/{duration:.4f}\\,1))"


def _zoom_and_pan(direction: str, cfg: RenderConfig) -> tuple[float, float, str, str]:
    """(zoom inicial, zoom final, fracao horizontal, fracao vertical).

    As fracoes posicionam a janela: 0 = colada na esquerda/topo, 1 = na
    direita/base, 0.5 = centro. `t` e `D` sao substituidos pelo chamador.
    """
    lo, hi = cfg.ken_burns.zoom_min, cfg.ken_burns.zoom_max
    if direction == "zoom_in":
        return lo, hi, "0.5", "0.5"
    if direction == "zoom_out":
        return hi, lo, "0.5", "0.5"
    if direction == "pan_right":
        return hi, hi, "RAMP_0_1", "0.5"
    if direction == "pan_left":
        return hi, hi, "RAMP_1_0", "0.5"
    raise ValueError(f"direcao de ken burns desconhecida: {direction!r}")


def fade_for(duration: float, cfg: RenderConfig) -> float:
    """Duracao do fade deste segmento. Encolhe junto com segmento curto.

    Os dois fades vivem dentro do proprio segmento — entrada em 0, saida
    terminando no fim — entao com fade fixo F o tempo em opacidade cheia e
    D - 2F, e abaixo de D = 2F os dois fades se sobrepoem e a imagem nunca
    chega a aparecer. Medido renderizando e lendo o brilho frame a frame,
    com imagem branca sobre base escura (base Y~55, branco cheio Y~234):

        D      fade fixo 0.4s              fade proporcional
        0.6s   pico Y=155, 0ms na tela     pico 234, 333ms
        0.8s   pico 234, 167ms             pico 234, 500ms
        1.2s   pico 234, 567ms             pico 234, 700ms

    Em 0.6s a imagem literalmente nao existe na tela: o pico fica em 2/3 do
    caminho e volta. Em 0.8s (exatamente 2F) ela toca o cheio num instante e
    ja sai — visualmente um piscar.

    `max_fade_ratio` poe um teto em F/D, garantindo metade do segmento em
    opacidade cheia qualquer que seja a duracao. Nao e clamp defensivo: e o
    que torna b-roll curto representavel, e sem ele `broll_min_seconds` teria
    um piso implicito de 2x o crossfade que nada no config revelaria.
    """
    return min(cfg.crossfade_seconds, max(0.0, duration) * cfg.max_fade_ratio)


def region_chain(region: Region) -> str:
    """Prefixo que recorta a regiao da tela preparada. Vazio no plano cheio.

    As fracoes saem de `iw`/`ih` e nao de numeros literais para o recorte
    continuar valendo se `prep_size` mudar — e `prep_size` e multiplo de 4
    justamente para que `iw/2` caia em pixel par e o crop nunca arredonde.
    """
    if region == "full":
        return ""
    if region not in SUB_SHOT_ORDER:
        raise ValueError(f"regiao de sub-plano desconhecida: {region!r}")
    x = "0" if region.endswith("_left") else "(iw-ow)"
    y = "0" if region.startswith("top_") else "(ih-oh)"
    return f"crop=iw/2:ih/2:{x}:{y},"


def canvas_for(region: Region, cfg: RenderConfig) -> tuple[int, int]:
    """Tela de trabalho do Ken Burns deste plano.

    No plano cheio e a tela 2x de sempre. No quadrante a fonte tem metade do
    tamanho em cada eixo, entao a tela tambem tem: com `canvas_scale: 2` o
    quadrante da exatamente a resolucao de saida, e o `scale` animado segue
    reduzindo. Manter a tela 2x aqui pediria um upscale de 2x de pixel que o
    `prep` ja interpolou uma vez — nao criaria detalhe nenhum, so amoleceria.
    """
    fraction = 1.0 if region == "full" else 0.5
    scale = cfg.ken_burns.canvas_scale * fraction
    return int(cfg.width * scale) // 2 * 2, int(cfg.height * scale) // 2 * 2


def ken_burns_chain(overlay: Overlay, cfg: RenderConfig) -> str:
    """Filtros de Ken Burns para uma imagem, sem os labels de entrada/saida."""
    canvas_w, canvas_h = canvas_for(overlay.region, cfg)
    region = region_chain(overlay.region)
    duration = overlay.duration

    z0, z1, fx, fy = _zoom_and_pan(overlay.direction, cfg)
    fx = _ramp(0.0, 1.0, duration) if fx == "RAMP_0_1" else (
        _ramp(1.0, 0.0, duration) if fx == "RAMP_1_0" else fx)

    if cfg.ken_burns.engine == "zoompan":
        return _zoompan_chain(overlay, cfg, z0, z1, region)

    # --- scale_crop (default) ---------------------------------------------
    if abs(z1 - z0) < 1e-9:
        # zoom constante (os pans): tamanho fixo, sem reconfiguracao por frame
        scale = f"scale={int(canvas_w * z0) // 2 * 2}:{int(canvas_h * z0) // 2 * 2}:flags=bicubic"
    else:
        zoom = _ramp(z0, z1, duration)
        scale = (
            f"scale=w='trunc({canvas_w}*{zoom}/2)*2':h='trunc({canvas_h}*{zoom}/2)*2'"
            f":eval=frame:flags=bicubic"
        )

    return (
        f"{region}{scale},"
        f"crop={canvas_w}:{canvas_h}:x='(iw-ow)*{fx}':y='(ih-oh)*{fy}',"
        f"scale={cfg.width}:{cfg.height}:flags=bicubic"
    )


def _zoompan_chain(overlay, cfg: RenderConfig, z0: float, z1: float,
                   region: str) -> str:
    """Escape hatch. Treme mais, mas nao depende de reconfiguracao por frame.

    O `select` na frente nao e decorativo: o `d` do zoompan conta frames de
    SAIDA por frame de ENTRADA, e a entrada aqui e uma imagem em `-loop 1`.
    Sem o select, um segmento de 12s a 30fps entrega 360 frames de entrada e
    o zoompan devolve 360 varreduras completas — 129.600 frames, com a
    imagem praticamente parada em cada trecho de 360.
    """
    frames = max(1, int(round(overlay.duration * cfg.fps)))
    zoom = f"{z0:.6f}+({z1 - z0:.6f})*on/{frames}"
    # O recorte da regiao vem DEPOIS do select: antes dele, ele rodaria em
    # cada frame de entrada do `-loop 1` para ser descartado em seguida.
    return (
        f"select='eq(n\\,0)',{region}"
        f"zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={frames}:s={cfg.width}x{cfg.height}:fps={cfg.fps:g}"
    )


# --------------------------------------------------------------------------
# filtergraph completo de um chunk
# --------------------------------------------------------------------------


def prep_size(cfg: RenderConfig) -> tuple[int, int]:
    """Tamanho da tela pre-renderizada de cada imagem.

    E a tela 2x multiplicada pelo zoom maximo, para que o `scale` animado
    sempre reduza — nunca amplie — e a imagem nao amoleca no zoom fechado.

    Arredonda para cima em multiplo de 4, e nao de 2, por causa do sub-plano:
    o quadrante e metade desta tela, e metade de um multiplo de 4 ainda e
    par. Sem isso, 1920x1080 com canvas 2x e zoom 1.12 daria 4302 e o
    quadrante sairia com 2151 — impar, e o `crop` teria que arredondar para
    baixo exatamente onde a margem para nao ampliar e de 0.6px (2150.4 e o
    que o Ken Burns pede). Com 4304, o quadrante e 2152 e sobra.
    """
    scale = cfg.ken_burns.canvas_scale * cfg.ken_burns.zoom_max
    return _up_to_four(cfg.width * scale), _up_to_four(cfg.height * scale)


def _up_to_four(value: float) -> int:
    return -(-int(ceil(value)) // 4) * 4


def video_trim_chain(keep: list[tuple[float, float]]) -> str:
    """Corta e reemenda os trechos da trilha de VIDEO, devolvendo `[cut]`.

    Os tempos precisam chegar ja alinhados ao grid de frames. `atrim` corta
    audio por amostra e `trim` corta video por frame: em tempo arbitrario,
    cada corte deixa ate um frame de diferenca entre as trilhas — medido,
    24ms de dessincronia acumulada em 20 cortes. Alinhado, zero.
    """
    n = len(keep)
    parts = "".join(
        f"[0:v]trim=start={a:.6f}:end={b:.6f},setpts=PTS-STARTPTS[t{i}];"
        for i, (a, b) in enumerate(keep)
    )
    return parts + "".join(f"[t{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[cut]"


def audio_trim_chain(keep: list[tuple[float, float]]) -> str:
    """O mesmo para a trilha de AUDIO, devolvendo `[aout]`.

    Roda num passe proprio, sem decodificar video: o audio e cortado e
    encodado exatamente uma vez, e os chunks de video seguem sendo `-an`.
    """
    n = len(keep)
    parts = "".join(
        f"[0:a]atrim=start={a:.6f}:end={b:.6f},asetpts=PTS-STARTPTS[s{i}];"
        for i, (a, b) in enumerate(keep)
    )
    return parts + "".join(f"[s{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[aout]"


def build_graph(
    chunk: Chunk, cfg: RenderConfig, subtitle_path: Path | None
) -> str:
    """Monta o filtergraph do chunk. Entrada 0 = video base; 1..N = overlays."""
    parts: list[str] = []

    if chunk.keep:
        parts.append(video_trim_chain(chunk.keep))
        base_source = "[cut]"
    else:
        base_source = "[0:v]"

    # base: normaliza resolucao, aspecto e cadencia
    parts.append(
        f"{base_source}scale={cfg.width}:{cfg.height}:flags=bicubic,setsar=1,"
        f"fps={cfg.fps:g},format=yuv420p[base0]"
    )

    current = "base0"

    for position, overlay in enumerate(chunk.overlays, start=1):
        label = f"b{position}"
        if overlay.image_path is None:
            visual = f"scale={cfg.width}:{cfg.height},setsar=1"
        else:
            visual = ken_burns_chain(overlay, cfg)

        # Um plano so deriva o fade da propria duracao; sub-plano recebe o
        # fade da faixa inteira nas pontas e 0.0 nas emendas internas. Fade de
        # duracao zero sai do grafo em vez de virar `d=0`: o que se quer ali e
        # corte seco, e um filtro a menos e uma coisa a menos para o ffmpeg
        # interpretar como "sem fade".
        derived = fade_for(overlay.duration, cfg)
        fade_in = derived if overlay.fade_in is None else overlay.fade_in
        fade_out = derived if overlay.fade_out is None else overlay.fade_out

        chain = [f"[{position}:v]{visual}", "setsar=1", "format=yuva420p"]
        if fade_in > 0:
            chain.append(f"fade=t=in:st=0:d={fade_in:g}:alpha=1")
        if fade_out > 0:
            chain.append(
                f"fade=t=out:st={max(0.0, overlay.duration - fade_out):.3f}"
                f":d={fade_out:g}:alpha=1"
            )
        chain.append(f"setpts=PTS+{overlay.start:.3f}/TB[{label}]")
        parts.append(",".join(chain))

        nxt = f"base{position}"
        parts.append(
            f"[{current}][{label}]overlay=0:0:format=auto:eof_action=pass"
            f":enable='between(t,{overlay.start:.3f},{overlay.end:.3f})'[{nxt}]"
        )
        current = nxt

    if subtitle_path is not None:
        escaped = str(subtitle_path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        parts.append(f"[{current}]subtitles='{escaped}'[vout]")
    else:
        parts.append(f"[{current}]null[vout]")

    return ";\n".join(parts)


def build_inputs(chunk: Chunk, source: Path, cfg: RenderConfig) -> list[str]:
    """Argumentos de entrada do ffmpeg, na ordem que o filtergraph espera.

    A janela lida da entrada nao e [start, end) do chunk quando ha corte:
    aqueles sao tempos da SAIDA, e a entrada precisa cobrir tambem o que vai
    ser descartado no meio.
    """
    window_start = chunk.input_start if chunk.input_start is not None else chunk.start
    window_duration = (
        chunk.input_duration if chunk.input_duration is not None else chunk.duration
    )
    args = ["-ss", f"{window_start:.3f}", "-t", f"{window_duration:.3f}", "-i", str(source)]
    for overlay in chunk.overlays:
        if overlay.image_path is None:
            args += [
                "-f", "lavfi", "-t", f"{overlay.duration:.3f}",
                "-i", f"color=c={cfg.solid_fallback_color}:s={cfg.width}x{cfg.height}:r={cfg.fps:g}",
            ]
        else:
            args += [
                "-loop", "1", "-framerate", f"{cfg.fps:g}",
                "-t", f"{overlay.duration:.3f}", "-i", str(overlay.image_path),
            ]
    return args


# --------------------------------------------------------------------------
# divisao em chunks
# --------------------------------------------------------------------------


def plan_chunks(
    overlays: list[Overlay],
    duration: float,
    cfg: RenderConfig,
    window: Callable[[float, float], tuple[float, float, list[tuple[float, float]]]] | None = None,
) -> list[Chunk]:
    """Divide a timeline em chunks cujo filtergraph cabe no limite do config.

    Os cortes caem no inicio de um overlay. Como os dois fades de alpha vivem
    dentro do proprio overlay, nenhuma transicao atravessa a fronteira de um
    chunk. Com sub-planos a fronteira pode cair dentro de uma faixa de
    b-roll, numa emenda entre dois planos da mesma imagem — e continua
    valendo, porque ali o corte ja e seco: nao ha fade nenhum para partir ao
    meio, e o ponto de concat coincide com um corte visual em vez de cair no
    meio de um movimento.
    """
    def finish(chunk: Chunk) -> Chunk:
        """Preenche a janela de entrada e os trechos a manter do chunk."""
        if window is not None:
            chunk.input_start, chunk.input_duration, chunk.keep = window(chunk.start, chunk.end)
        return chunk

    if not overlays:
        return _warn_if_over([finish(Chunk(index=0, start=0.0, end=duration))], cfg)

    single = finish(Chunk(index=0, start=0.0, end=duration, overlays=_localize(overlays, 0.0)))
    if len(build_graph(single, cfg, Path("subs.ass"))) <= cfg.max_filtergraph_chars:
        return [single]

    chunks: list[Chunk] = []
    batch: list[Overlay] = []
    chunk_start = 0.0

    for overlay in overlays:
        candidate = batch + [overlay]
        probe = finish(Chunk(index=len(chunks), start=chunk_start,
                             end=candidate[-1].end,
                             overlays=_localize(candidate, chunk_start)))
        too_long = len(build_graph(probe, cfg, Path("subs.ass"))) > cfg.max_filtergraph_chars
        if too_long and batch:
            chunks.append(finish(Chunk(
                index=len(chunks), start=chunk_start, end=overlay.start,
                overlays=_localize(batch, chunk_start))))
            chunk_start = overlay.start
            batch = [overlay]
        else:
            batch = candidate

    chunks.append(finish(Chunk(
        index=len(chunks), start=chunk_start, end=duration,
        overlays=_localize(batch, chunk_start))))
    return _warn_if_over(chunks, cfg)


def _warn_if_over(chunks: list[Chunk], cfg: RenderConfig) -> list[Chunk]:
    """Avisa quando um chunk nao caber no limite, em vez de deixar o ffmpeg falar.

    As fronteiras de chunk sao as fronteiras de b-roll, entao o grafo de um
    chunk cresce com o que estiver DENTRO dele — e com o aperto de pausa
    ligado isso e o numero de emendas, que escala com a duracao e nao com o
    numero de b-rolls. Um trecho longo de a-roll corrido com aperto agressivo
    nao tem onde ser partido.

    Nao ha fallback automatico porque nao existe um bom: partir no meio de
    uma emenda mudaria o corte, e o que resolve de verdade e subir o
    `max_filtergraph_chars` ou afrouxar o
    `pause_max_seconds`. O aviso diz qual dos dois.
    """
    for chunk in chunks:
        size = len(build_graph(chunk, cfg, Path("subs.ass")))
        if size > cfg.max_filtergraph_chars:
            log("render.warn",
                chunk=chunk.index, chars=size, limite=cfg.max_filtergraph_chars,
                emendas=len(chunk.keep),
                detail="filtergraph acima do limite; suba render.max_filtergraph_chars "
                       "ou afrouxe trim.pause_max_seconds")
    return chunks


def _localize(overlays: list[Overlay], chunk_start: float) -> list[Overlay]:
    """Reposiciona os overlays para o tempo local do chunk."""
    return [
        Overlay(start=o.start - chunk_start, end=o.end - chunk_start,
                direction=o.direction, image_path=o.image_path, region=o.region,
                fade_in=o.fade_in, fade_out=o.fade_out)
        for o in overlays
    ]
