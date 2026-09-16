"""Smoke test do render, executando ffmpeg de verdade.

Os testes de `test_filtergraph.py` cobrem o texto do grafo; estes cobrem o
que so a execucao revela. Rodam em 640x360 com preset ultrafast porque as
propriedades sob teste — duracao exata, audio intacto, chunks que ladrilham a
timeline, legenda deslocada por chunk — nao dependem da resolucao.

Pula inteiro se ffmpeg nao estiver no PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from pipeline.schemas import (
    EDL,
    AssetEstimate,
    AssetItem,
    Assets,
    EDLSegment,
    EDLStats,
    Transcript,
    TranscriptSegment,
    Word,
)
from pipeline.stages import ingest, render

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe nao estao no PATH",
)

DURATION = 24.0
SEG = 3.0
N_SEGMENTS = 8


def ffprobe(path: Path, entries: str) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", entries, "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def audio_md5(path: Path) -> str:
    """md5 do stream de audio COPIADO, sem decodificar.

    Se o pipeline tiver reencodado o audio em qualquer ponto, este valor muda.
    """
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a", "-c", "copy", "-f", "md5", "-"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("media")
    source = root / "source.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=s=640x360:r=30:d={DURATION:g}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={DURATION:g}",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
         "-c:a", "aac", "-b:a", "64k", "-shortest", str(source)],
        check=True, capture_output=True,
    )
    image = root / "broll.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "gradients=s=800x450:c0=0x101820:c1=0xE8B04B",
         "-frames:v", "1", str(image)],
        check=True, capture_output=True,
    )
    return {"root": root, "source": source, "image": image}


@pytest.fixture
def smoke_config(config, media, tmp_path):
    """Config de teste: saida pequena, encode rapido, work isolado."""
    cfg = config.model_copy(deep=True)
    cfg.root = tmp_path
    cfg.render.width, cfg.render.height = 640, 360
    cfg.render.preset = "ultrafast"
    cfg.render.crf = 30
    cfg.subtitles.font_name = "DejaVu Sans"
    shutil.copy(media["image"], tmp_path / "broll.png")
    return cfg


def make_artifacts(smoke_config, media, *, n_broll: int):
    """manifest real (via ingest) + transcript/edl/assets montados na mao."""
    manifest = ingest.run(media["source"], smoke_config)

    segments = []
    for i in range(N_SEGMENTS):
        start, end = i * SEG, (i + 1) * SEG
        tokens = f"segmento numero {i} do transcript".split()
        step = (end - start) / len(tokens)
        segments.append(TranscriptSegment(
            id=i, start=start, end=end, text=" ".join(tokens),
            words=[Word(start=round(start + j * step, 3),
                        end=round(start + (j + 1) * step, 3), word=t)
                   for j, t in enumerate(tokens)],
        ))
    transcript = Transcript(input_hash=manifest.audio_hash, language="pt",
                            model="test", duration=DURATION, segments=segments)

    # aroll[0..1] broll[2..3] aroll[4] broll[5..7], ou so o primeiro broll
    layout = [("aroll", 0, 1), ("broll", 2, 3), ("aroll", 4, 4), ("broll", 5, 7)]
    if n_broll == 1:
        layout = [("aroll", 0, 1), ("broll", 2, 3), ("aroll", 4, 7)]

    edl_segments = [
        EDLSegment(kind=kind, first_segment=first, last_segment=last,
                   concept="a warm gradient" if kind == "broll" else "",
                   concept_tags=["gradient", "warm"] if kind == "broll" else [],
                   start=first * SEG, end=(last + 1) * SEG)
        for kind, first, last in layout
    ]
    broll_seconds = sum(s.end - s.start for s in edl_segments if s.kind == "broll")
    edl = EDL(
        input_hash=transcript.digest(), model="test", attempts=1, duration=DURATION,
        segments=edl_segments,
        stats=EDLStats(broll_ratio=broll_seconds / DURATION, switches_per_minute=1.0,
                       switches_per_minute_max=2.0,
                       n_aroll=sum(1 for s in edl_segments if s.kind == "aroll"),
                       n_broll=sum(1 for s in edl_segments if s.kind == "broll"),
                       broll_seconds=broll_seconds),
    )
    items = [
        AssetItem(segment_index=i, origin="generated", path="broll.png",
                  prompt="a warm gradient", cost_usd=0.003)
        for i, s in enumerate(edl_segments) if s.kind == "broll"
    ]
    assets = Assets(
        input_hash=edl.digest(), items=items,
        estimate=AssetEstimate(n_broll=len(items), n_from_bank=0, n_from_stock=0,
                               n_to_generate=len(items),
                               worst_case_usd=0.003 * len(items),
                               estimated_usd=0.003 * len(items),
                               budget_usd=1.5, within_budget=True),
        total_cost_usd=0.003 * len(items), by_origin={"generated": len(items)},
    )
    return manifest, transcript, edl, assets


# --------------------------------------------------------------------------
# passe unico
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _single_pass_cache():
    return {}


def run_render(smoke_config, media, *, n_broll=2, max_chars=3000):
    smoke_config.render.max_filtergraph_chars = max_chars
    manifest, transcript, edl, assets = make_artifacts(smoke_config, media, n_broll=n_broll)
    output = render.run(manifest, transcript, edl, assets, smoke_config)
    return manifest, output


def test_passe_unico_preserva_a_duracao(smoke_config, media):
    manifest, output = run_render(smoke_config, media)
    assert float(ffprobe(output, "format=duration")) == pytest.approx(DURATION, abs=0.05)


def test_passe_unico_nao_toca_no_audio(smoke_config, media):
    """A propriedade central do pipeline: o audio que sai e o que entrou."""
    manifest, output = run_render(smoke_config, media)
    assert audio_md5(output) == audio_md5(media["source"])


def test_saida_tem_a_resolucao_do_config(smoke_config, media):
    manifest, output = run_render(smoke_config, media)
    assert ffprobe(output, "stream=width,height").startswith("640,360")


def test_filtergraph_fica_em_disco_para_depurar(smoke_config, media):
    manifest, output = run_render(smoke_config, media)
    graph = (smoke_config.work_dir / manifest.slug / "filtergraph.txt").read_text()
    assert "overlay" in graph and "[vout]" in graph
    assert "xfade" not in graph


def _frame_rgb(video: Path, timestamp: float) -> bytes:
    """Metade de cima do frame, em 32x18 rgb24. Metade de cima para ficar
    fora da legenda, que e queimada igual nos dois casos."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{timestamp:g}", "-i", str(video),
         "-frames:v", "1", "-vf", "crop=iw:ih/2:0:0,scale=32:18",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    )
    return out.stdout


def _mean_abs_diff(a: bytes, b: bytes) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / max(1, min(len(a), len(b)))


def test_broll_cobre_a_imagem_e_aroll_nao(smoke_config, media):
    """Compara o frame final com o frame do ORIGINAL no mesmo instante.

    Em a-roll os dois tem que ser praticamente o mesmo quadro; em b-roll o
    original tem que ter sido coberto. E a definicao operacional de "partes
    da minha imagem sao cobertas por b-roll".
    """
    manifest, output = run_render(smoke_config, media)
    source = media["source"]

    # aroll em 3.0s (faixa 0-6s), broll em 9.0s (faixa 6-12s, longe dos fades)
    aroll_diff = _mean_abs_diff(_frame_rgb(output, 3.0), _frame_rgb(source, 3.0))
    broll_diff = _mean_abs_diff(_frame_rgb(output, 9.0), _frame_rgb(source, 9.0))

    assert aroll_diff < 12, f"o a-roll foi alterado (diff {aroll_diff:.1f})"
    assert broll_diff > 40, f"o b-roll nao cobriu o original (diff {broll_diff:.1f})"


def test_crossfade_e_um_estado_intermediario(smoke_config, media):
    """No meio dos 400ms de fade-in o quadro nao e nem o a-roll nem o b-roll
    cheio: e uma mistura dos dois."""
    manifest, output = run_render(smoke_config, media)
    source = media["source"]
    fade = smoke_config.render.crossfade_seconds

    at_start = _mean_abs_diff(_frame_rgb(output, 6.02), _frame_rgb(source, 6.02))
    mid_fade = _mean_abs_diff(_frame_rgb(output, 6.0 + fade / 2), _frame_rgb(source, 6.0 + fade / 2))
    full = _mean_abs_diff(_frame_rgb(output, 9.0), _frame_rgb(source, 9.0))

    assert at_start < mid_fade < full, (at_start, mid_fade, full)


# --------------------------------------------------------------------------
# chunks + concat
# --------------------------------------------------------------------------


def test_chunks_preservam_a_duracao(smoke_config, media):
    manifest, output = run_render(smoke_config, media, max_chars=600)
    chunks = sorted((smoke_config.work_dir / manifest.slug / "chunks").glob("chunk_*.mp4"))
    assert len(chunks) > 1, "o limite baixo devia ter forcado o caminho de chunks"
    assert float(ffprobe(output, "format=duration")) == pytest.approx(DURATION, abs=0.05)


def test_chunks_nao_tocam_no_audio(smoke_config, media):
    manifest, output = run_render(smoke_config, media, max_chars=600)
    assert audio_md5(output) == audio_md5(media["source"])


def test_duracoes_dos_chunks_somam_a_duracao_original(smoke_config, media):
    """Se nao somarem exatamente, o audio (stream copy) deriva."""
    manifest, output = run_render(smoke_config, media, max_chars=600)
    chunks = sorted((smoke_config.work_dir / manifest.slug / "chunks").glob("chunk_*.mp4"))
    total = sum(float(ffprobe(c, "format=duration")) for c in chunks)
    assert total == pytest.approx(DURATION, abs=0.05)


def test_legenda_e_deslocada_por_chunk(smoke_config, media):
    """A legenda do chunk 1 comeca em zero, nao no tempo global.

    Liga a legenda explicitamente: o config de exemplo a deixa desligada, e
    um teste do comportamento da legenda nao pode depender desse default.
    """
    smoke_config.subtitles.enabled = True
    manifest, output = run_render(smoke_config, media, max_chars=600)
    work = smoke_config.work_dir / manifest.slug
    second = (work / "subs_001.ass").read_text()
    rows = [r for r in second.splitlines() if r.startswith("Dialogue")]
    assert rows, "o segundo chunk saiu sem legenda"
    assert rows[0].split(",")[1] == "0:00:00.00"


def test_um_filtergraph_por_chunk_em_disco(smoke_config, media):
    manifest, output = run_render(smoke_config, media, max_chars=600)
    work = smoke_config.work_dir / manifest.slug
    graphs = list(work.glob("filtergraph*.txt"))
    chunks = list((work / "chunks").glob("chunk_*.mp4"))
    assert len(graphs) == len(chunks)
    for graph in graphs:
        assert len(graph.read_text()) <= 600 + 200  # o texto em disco tem os \n


# --------------------------------------------------------------------------
# fallback de cor solida
# --------------------------------------------------------------------------


def test_segmento_sem_imagem_renderiza_cor_solida(smoke_config, media):
    manifest, transcript, edl, assets = make_artifacts(smoke_config, media, n_broll=2)
    for item in assets.items:
        item.origin, item.path = "solid", None
    assets.by_origin = {"solid": len(assets.items)}
    output = render.run(manifest, transcript, edl, assets, smoke_config)

    assert float(ffprobe(output, "format=duration")) == pytest.approx(DURATION, abs=0.05)
    assert audio_md5(output) == audio_md5(media["source"])


# --------------------------------------------------------------------------
# corte seco
# --------------------------------------------------------------------------


def test_corte_preserva_sincronia_audio_video(smoke_config, media):
    """A propriedade que o corte nao pode quebrar.

    `atrim` corta audio por amostra e `trim` corta video por frame: em tempo
    nao alinhado, cada corte deixa ate um frame de diferenca. Medido com 20
    cortes: 24ms de deriva. Os trechos mantidos vem alinhados ao grid de
    frames justamente para isso dar zero.
    """
    from pipeline.filtergraph import Chunk, audio_trim_chain, build_graph, build_inputs
    from pipeline.schemas import KeepRange, TrimPlan, TrimStats

    fps = smoke_config.render.fps
    keep = [(round(a * fps) / fps, round(b * fps) / fps)
            for a, b in [(0, 4), (5.5, 12), (13.7, DURATION)]]
    trimmed = sum(b - a for a, b in keep)

    plan = TrimPlan(
        input_hash="h", enabled=True, cuts=[],
        keep=[KeepRange(start=a, end=b) for a, b in keep],
        stats=TrimStats(original_seconds=DURATION, trimmed_seconds=trimmed,
                        removed_seconds=DURATION - trimmed,
                        removed_ratio=(DURATION - trimmed) / DURATION,
                        n_cuts=2, n_pause_cuts=2, n_filler_cuts=0),
    )

    manifest, transcript, edl, assets = make_artifacts(smoke_config, media, n_broll=1)
    work = smoke_config.work_dir / manifest.slug
    work.mkdir(parents=True, exist_ok=True)

    chunk = Chunk(0, 0.0, trimmed, [], keep=plan.keep_within(0.0, trimmed),
                  input_start=0.0, input_duration=DURATION)
    graph = build_graph(chunk, smoke_config.render, None)

    video = work / "tv.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         *build_inputs(chunk, media["source"], smoke_config.render),
         "-filter_complex", graph.replace(";\n", ";"), "-map", "[vout]", "-an",
         "-c:v", "libx264", "-crf", "35", "-preset", "ultrafast",
         "-r", f"{fps:g}", str(video)],
        check=True, capture_output=True,
    )
    audio = work / "ta.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(media["source"]), "-vn",
         "-filter_complex", audio_trim_chain(keep), "-map", "[aout]",
         "-c:a", "aac", "-b:a", "192k", str(audio)],
        check=True, capture_output=True,
    )
    final = work / "cut.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(audio),
         "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-shortest", str(final)],
        check=True, capture_output=True,
    )

    dur_v = float(ffprobe(final, "stream=duration").splitlines()[0])
    dur_a = float(ffprobe(final, "stream=duration").splitlines()[1])

    assert dur_v == pytest.approx(trimmed, abs=0.05), f"video {dur_v} != {trimmed}"
    assert dur_a == pytest.approx(trimmed, abs=0.05), f"audio {dur_a} != {trimmed}"
    assert abs(dur_v - dur_a) < 0.02, f"dessincronia de {abs(dur_v - dur_a) * 1000:.0f}ms"


def test_dezenas_de_emendas_sobrevivem_ao_filtergraph(smoke_config, media):
    """O custo real do aperto de pausa: muitos pontos de corte num grafo so.

    O aperto troca 3 cortes por algumas dezenas, e cada um vira um par
    `trim`/`setpts` mais uma entrada no concat. Este teste renderiza de fato,
    porque o que pode quebrar aqui — estouro do grafo, deriva acumulada entre
    as trilhas — nao aparece em teste de string.
    """
    from pipeline.config import TrimConfig
    from pipeline.filtergraph import Chunk, audio_trim_chain, build_graph, build_inputs
    from pipeline.schemas import Transcript, TranscriptSegment, Word
    from pipeline.trim import build_plan

    fps = smoke_config.render.fps

    # fala corrida: palavra de 0.45s a cada 0.75s, sem nenhuma pausa longa
    words, tempo = [], 0.0
    while tempo + 0.45 < DURATION:
        words.append(Word(start=round(tempo, 3), end=round(tempo + 0.45, 3), word="pa"))
        tempo += 0.75
    transcript = Transcript(
        input_hash="h", language="pt", model="small", duration=DURATION,
        segments=[TranscriptSegment(id=0, start=0.0, end=words[-1].end,
                                    text="pa " * len(words), words=words)],
    )

    plan = build_plan(transcript, TrimConfig(pause_max_seconds=0.12), fps)
    assert plan.stats.n_squeeze_cuts > 20, plan.stats.model_dump()
    assert plan.stats.n_pause_cuts == 0        # nenhuma pausa longa nesta fala
    trimmed = sum(r.duration for r in plan.keep)

    manifest, transcript_a, edl, assets = make_artifacts(smoke_config, media, n_broll=1)
    work = smoke_config.work_dir / manifest.slug
    work.mkdir(parents=True, exist_ok=True)

    chunk = Chunk(0, 0.0, trimmed, [], keep=plan.keep_within(0.0, trimmed),
                  input_start=0.0, input_duration=DURATION)
    graph = build_graph(chunk, smoke_config.render, None)

    video = work / "muitas.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         *build_inputs(chunk, media["source"], smoke_config.render),
         "-filter_complex", graph.replace(";\n", ";"), "-map", "[vout]", "-an",
         "-c:v", "libx264", "-crf", "35", "-preset", "ultrafast",
         "-r", f"{fps:g}", str(video)],
        check=True, capture_output=True,
    )
    audio = work / "muitas.m4a"
    keep_pairs = [(r.start, r.end) for r in plan.keep]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(media["source"]), "-vn",
         "-filter_complex", audio_trim_chain(keep_pairs), "-map", "[aout]",
         "-c:a", "aac", "-b:a", "192k", str(audio)],
        check=True, capture_output=True,
    )

    dur_v = float(ffprobe(video, "format=duration"))
    dur_a = float(ffprobe(audio, "format=duration"))
    assert dur_v == pytest.approx(trimmed, abs=0.08), f"video {dur_v} != {trimmed}"
    # a deriva que o alinhamento ao grid de frames existe para evitar, agora
    # com uma ordem de magnitude mais de cortes para acumular
    assert abs(dur_v - dur_a) < 0.05, f"dessincronia de {abs(dur_v - dur_a) * 1000:.0f}ms"


def test_broll_curto_chega_a_opacidade_cheia(smoke_config, media):
    """A propriedade que o fade proporcional existe para garantir.

    Mede 0.6s, nao 0.8s, e conta FRAMES na tela em vez de amostrar um ponto.
    As duas escolhas sao o que fazem o teste discriminar: em 0.8s (exatamente
    2x o crossfade) o fade fixo ainda toca a opacidade cheia num instante, e
    uma amostra no apice passa nos dois comportamentos. Em 0.6s o fade fixo
    tem pico Y=155 e zero frame na tela.
    """
    from pipeline.filtergraph import Chunk, Overlay, build_graph, build_inputs

    branca = smoke_config.work_dir / "branca.png"
    branca.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "color=c=white:s=1920x1080:d=1", "-frames:v", "1", str(branca)],
        check=True, capture_output=True,
    )

    inicio, duracao = 4.0, 0.6
    overlay = Overlay(start=inicio, end=inicio + duracao,
                      direction="zoom_in", image_path=branca)
    chunk = Chunk(0, 0.0, DURATION, [overlay], keep=[(0.0, DURATION)],
                  input_start=0.0, input_duration=DURATION)

    saida = smoke_config.work_dir / "curto.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         *build_inputs(chunk, media["source"], smoke_config.render),
         "-filter_complex", build_graph(chunk, smoke_config.render, None).replace(";\n", ";"),
         "-map", "[vout]", "-an",
         "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast", str(saida)],
        check=True, capture_output=True,
    )

    fps = smoke_config.render.fps
    brilhos = []
    for indice in range(int(duracao * fps)):
        em = inicio + indice / fps + 1 / (2 * fps)
        quadro = smoke_config.work_dir / "q.png"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{em:.4f}", "-i", str(saida),
             "-frames:v", "1", "-vf", "scale=1:1", str(quadro)],
            check=True, capture_output=True,
        )
        medida = subprocess.run(
            ["ffprobe", "-v", "error", "-f", "lavfi",
             "-i", f"movie={quadro.as_posix()},signalstats",
             "-show_entries", "frame_tags=lavfi.signalstats.YAVG",
             "-of", "default=nw=1:nk=1"],
            check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()[0]
        brilhos.append(float(medida))

    cheios = [y for y in brilhos if y > 200]
    assert max(brilhos) > 220, f"pico Y={max(brilhos):.0f}: a imagem nunca apareceu"
    # metade do segmento em opacidade cheia, com folga para o encode
    assert len(cheios) >= len(brilhos) * 0.4, (
        f"{len(cheios)}/{len(brilhos)} frames na tela; perfil={[round(y) for y in brilhos]}"
    )


def _luma(video: Path, at: float) -> int:
    """Brilho medio do frame em `at`, em um byte.

    `scale=1:1:flags=area` e media de caixa sobre o frame inteiro — nao uma
    amostra de um pixel, que num plano com movimento cairia em lugar
    diferente a cada frame.
    """
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{at:.4f}", "-i", str(video),
         "-frames:v", "1", "-vf", "scale=1:1:flags=area",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    )
    return out.stdout[0]


def test_sub_planos_mostram_regioes_diferentes_da_mesma_imagem(smoke_config, media):
    """A propriedade central do sub-plano, medida em vez de argumentada.

    A imagem tem os quatro quadrantes em tons distintos, entao o brilho medio
    do frame DIZ qual regiao esta na tela. Medido:

        full          Y=122      (ve os quatro quadrantes)
        top_left      Y= 17
        bottom_right  Y=242
        top_right     Y= 93

    O teste discrimina — verificado forcando `region="full"` nos quatro
    planos, o que da Y=122/129/136/129. Ou seja: sem o recorte, os quatro
    planos ficam dentro de 14 pontos um do outro E dentro de 8 pontos do
    a-roll (Y=130). Nem este teste nem o de cobertura de b-roll veriam a
    diferenca por outro caminho; e o brilho por regiao que a ve.
    """
    from pipeline.filtergraph import Chunk, build_graph, build_inputs, prep_size

    largura, altura = prep_size(smoke_config.render)
    imagem = smoke_config.work_dir / "quadrantes.png"
    imagem.parent.mkdir(parents=True, exist_ok=True)
    cantos = [(0, 0), (largura // 2, 0), (0, altura // 2), (largura // 2, altura // 2)]
    caixas = ",".join(
        f"drawbox=x={x}:y={y}:w={largura // 2}:h={altura // 2}"
        f":color=0x{tom:02X}{tom:02X}{tom:02X}:t=fill"
        for (x, y), tom in zip(cantos, [0x20, 0x60, 0xA0, 0xE0])
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", f"color=c=black:s={largura}x{altura}", "-vf", caixas,
         "-frames:v", "1", str(imagem)],
        check=True, capture_output=True,
    )

    inicio, fim = 6.0, 16.0        # 10s = 4 planos de 2.5s
    edl_um_broll = EDL(
        input_hash="x", model="test", attempts=1, duration=DURATION,
        segments=[EDLSegment(kind="broll", first_segment=0, last_segment=0,
                             start=inicio, end=fim)],
        stats=EDLStats(broll_ratio=(fim - inicio) / DURATION, switches_per_minute=1.0,
                       switches_per_minute_max=1.0, n_aroll=0, n_broll=1,
                       broll_seconds=fim - inicio),
    )
    planos = render.build_overlays(edl_um_broll, {0: imagem}, smoke_config)
    assert [p.region for p in planos] == ["full", "top_left", "bottom_right", "top_right"]

    chunk = Chunk(0, 0.0, DURATION, planos)
    saida = smoke_config.work_dir / "sub_planos.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         *build_inputs(chunk, media["source"], smoke_config.render),
         "-filter_complex", build_graph(chunk, smoke_config.render, None).replace(";\n", ";"),
         "-map", "[vout]", "-an",
         "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast", str(saida)],
        check=True, capture_output=True,
    )

    medido = {p.region: _luma(saida, (p.start + p.end) / 2) for p in planos}
    perfil = {r: int(y) for r, y in medido.items()}

    # cada plano mostra a SUA regiao: a ordem dos tons da imagem, preservada
    assert medido["top_left"] < medido["top_right"] < medido["bottom_right"], perfil
    # o plano cheio ve os quatro quadrantes, entao cai no meio — nao e nenhum
    assert medido["top_right"] < medido["full"] < medido["bottom_right"], perfil
    # separacao bem acima dos 14 pontos que o caminho sem recorte produz
    valores = sorted(medido.values())
    assert min(b - a for a, b in zip(valores, valores[1:])) > 20, perfil


def test_imagem_ilegivel_vira_cor_solida_em_vez_de_derrubar(smoke_config, media):
    """Download interrompido deixa um arquivo que existe, tem bytes, e nao e
    imagem. O `ffprobe` devolve `width: 0` em vez de falhar, e o render
    morria com ZeroDivisionError no aviso de resolucao — depois de as imagens
    ja terem sido pagas, por causa de uma so.

    Achado rodando o caminho inteiro pela CLI, nao por leitura do diff.
    """
    manifest, transcript, edl, assets = make_artifacts(smoke_config, media, n_broll=1)
    quebrada = smoke_config.root / "quebrada.png"
    quebrada.write_bytes(b"\x89PNG\r\n\x1a\n isto nao e uma imagem")
    for item in assets.items:
        item.path = "quebrada.png"

    output = render.run(manifest, transcript, edl, assets, smoke_config)

    assert float(ffprobe(output, "format=duration")) == pytest.approx(DURATION, abs=0.05)
    assert audio_md5(output) == audio_md5(media["source"])
