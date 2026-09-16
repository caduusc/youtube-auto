"""O caminho roteiro-primeiro inteiro, pela CLI de verdade.

Este arquivo existe por uma razao especifica: os testes de unidade passaram
todos enquanto a CLI imprimia "2 imagemns". O que so a execucao real revela e
o que esta aqui — argumento que nao chega, `f-string` errada, artefato gravado
com nome que o comando seguinte nao le, comando que existe no parser e nao no
dispatch.

Roda `main()` com `argv`, como o terminal faz. O provider de imagem e o banco
sao dublês (nenhum centavo, nenhum torch); o `ffmpeg` e real; a transcricao e
plantada em disco para o `faster-whisper` nao ser baixado — o que o teste
verifica dela e que o estagio LE o que esta la, que e justamente o mecanismo
de retomada.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from test_bank_reuse import FakeEmbedder
from test_images_stage import install_fakes

from pipeline import approval
from pipeline.cli import main
from pipeline.filtergraph import prep_size
from pipeline.script import parse as parse_script
from pipeline.stages.storyboard import cache_key as storyboard_key
from pipeline.schemas import (
    Assets, Placements, Storyboard, StoryboardBeat, Transcript,
    TranscriptSegment, Word,
)
from pipeline.util import read_json, write_json

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe nao estao no PATH",
)

DURACAO = 30.0
SEG = 2.5
N = 12                      # 12 segmentos de 2.5s

FALAS = [
    "todo mundo acha que precisa de uma equipe inteira para isso",
    "um editor alguem para escolher as imagens alguem para revisar",
    "hoje eu gravo falando quinze minutos e o resto acontece sozinho",
    "o custo nao esta na camera e nunca esteve",
    "esta no tempo que a edicao come todos os dias",
    "e tempo e a unica coisa que voce nao compra de volta",
    "a maquina nao tem gosto mas tem paciencia infinita",
    "ela le a transcricao inteira e acha onde a fala pede imagem",
    "ela nao vai escolher melhor que voce vai escolher razoavelmente bem",
    "oitenta vezes por video sem reclamar nenhuma vez",
    "e no fim do mes a conta inteira cabe num almoco de domingo",
    "e isso muda o que da para fazer sozinho",
]

ROTEIRO = """\
---
slug: esteira-de-video
subject: como montar uma esteira de producao de video sozinho
argument: a automacao ja e barata o bastante para sustentar um canal
viewer_takeaway: da para montar o estudio inteiro em software
duration_target_seconds: 30.0
---

## beat 1 — abertura
{0} {1} {2}

## beat 2 — onde o custo esta
{3} {4} {5}

## beat 3 — o que a maquina faz bem
{6} {7} {8}
""".format(*FALAS)


def um_board(script, cfg) -> Storyboard:
    """Ancoras copiados LITERALMENTE de falas que vao estar no transcript.

    E o caso comum e o que o desenho otimiza: seguindo o roteiro, a busca
    literal acerta e o embedding nunca e tocado.
    """
    ancoras = {
        1: "precisa de uma equipe inteira",
        2: "o custo nao esta na camera",
        3: "tem paciencia infinita",
    }
    conceitos = {
        1: ("an empty editing suite at night with three dark monitors", ["suite", "monitors"]),
        2: ("a cinema camera unused on a shelf beside invoices", ["camera", "shelf"]),
        3: ("a long workbench with eighty identical wooden spools", ["bench", "spools"]),
    }
    return Storyboard(input_hash=storyboard_key(script, cfg), beats=[
        StoryboardBeat(
            beat_id=i, script_anchor=ancoras[i],
            concept=conceitos[i][0], concept_tags=conceitos[i][1],
            sub_shots=2, rationale="porque o argumento pede imagem concreta",
            anchor_offset=0, seconds_per_shot=2.5,
        )
        for i in (1, 2, 3)
    ])


def um_transcript(audio_hash: str) -> Transcript:
    segments = []
    for i, texto in enumerate(FALAS):
        start, end = i * SEG, (i + 1) * SEG
        tokens = texto.split()
        passo = (end - start) / len(tokens)
        segments.append(TranscriptSegment(
            id=i, start=start, end=end, text=texto,
            words=[Word(start=round(start + j * passo, 3),
                        end=round(start + (j + 1) * passo, 3), word=w)
                   for j, w in enumerate(tokens)],
        ))
    return Transcript(input_hash=audio_hash, language="pt", model="plantado",
                      duration=DURACAO, segments=segments)


class ProviderComPng:
    """Gera um PNG de verdade, do tamanho que o prep pede.

    O `FakeProvider` dos outros testes escreve bytes que nao sao imagem, e
    para o render isso agora e "imagem ilegivel, virou cor solida" — o que
    faria este teste passar sem exercitar prep, Ken Burns nem sub-plano.
    """

    cost_usd_per_image = 0.025

    def __init__(self, size: tuple[int, int]) -> None:
        self.calls: list[str] = []
        self.width, self.height = size

    def generate(self, prompt: str, destination) -> float:
        self.calls.append(prompt)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tom = 0x20 + 0x40 * (len(self.calls) % 4)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
             "-i", f"gradients=s={self.width}x{self.height}"
                   f":c0=0x{tom:02X}1820:c1=0xE8B04B",
             "-frames:v", "1", str(destination)],
            check=True, capture_output=True,
        )
        return self.cost_usd_per_image


@pytest.fixture(scope="module")
def gravacao(tmp_path_factory) -> Path:
    """Um arquivo de video com audio, como o que sai da camera."""
    raiz = tmp_path_factory.mktemp("gravacao")
    arquivo = raiz / "tomada.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=s=640x360:r=30:d={DURACAO:g}",
         "-f", "lavfi", "-i", f"sine=frequency=330:duration={DURACAO:g}",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
         "-c:a", "aac", "-b:a", "64k", "-shortest", str(arquivo)],
        check=True, capture_output=True,
    )
    return arquivo


@pytest.fixture
def projeto(config, tmp_path, monkeypatch):
    """Um `work/` com o roteiro e o storyboard prontos, e um config.yaml.

    O roteiro e o storyboard entram em disco em vez de vir de chamada de
    Opus: o que esta sob teste aqui e a costura dos estagios, e os dois
    agentes ja tem teste proprio com a API dublada.
    """
    cfg = config.model_copy(deep=True)
    cfg.root = tmp_path
    cfg.render.width, cfg.render.height = 640, 360
    cfg.render.preset, cfg.render.crf = "ultrafast", 30
    cfg.subtitles.font_name = "DejaVu Sans"
    cfg.editorial.intro_aroll_seconds = 5.0      # video de 30s, nao de 15 min

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        Path("config.example.yaml").read_text(encoding="utf-8")
        .replace("width: 1920", "width: 640")
        .replace("height: 1080", "height: 360")
        .replace("preset: medium", "preset: ultrafast")
        .replace("intro_aroll_seconds: 20", "intro_aroll_seconds: 5"),
        encoding="utf-8",
    )

    work = tmp_path / "work" / "esteira-de-video"
    work.mkdir(parents=True)
    (work / "script.md").write_text(ROTEIRO, encoding="utf-8")
    # o storyboard entra em disco com o `input_hash` que o estagio 2 daria,
    # calculado com o config QUE A CLI VAI CARREGAR
    from pipeline.config import load_config
    write_json(work / "storyboard.json",
               um_board(parse_script(ROTEIRO), load_config(config_path)))

    provider = ProviderComPng(prep_size(cfg.render))
    install_fakes(monkeypatch, cfg, provider=provider)
    # o estagio 3 tem `open_bank` proprio; o align usa o embedder direto
    monkeypatch.setattr(
        "pipeline.stages.align.SentenceTransformerEmbedder",
        lambda *a, **k: FakeEmbedder())
    return config_path, work, provider


def cli(config_path: Path, *args: str) -> int:
    return main(["--config", str(config_path), *args])


# --------------------------------------------------------------------------


def test_nao_gasta_sem_o_storyboard_aprovado(projeto):
    """O portao que o estagio 3 confere e o do storyboard: e a ultima coisa
    que voce leu antes de gastar. A mensagem diz o comando exato."""
    config_path, work, provider = projeto

    # SystemExit e nao Exception: portao fechado tem que virar mensagem para
    # uma pessoa, nao traceback. Com `Exception` este teste passava nos dois.
    with pytest.raises(SystemExit, match="pipeline approve storyboard"):
        cli(config_path, "images", "esteira-de-video")
    assert provider.calls == []


def test_editar_o_roteiro_depois_do_storyboard_nao_deixa_gastar(projeto):
    """O buraco que o portao sozinho nao fecha.

    Aprovar o roteiro, gerar o storyboard, aprovar o storyboard e DEPOIS
    editar o roteiro deixa os dois portoes de pe — e as imagens ilustrariam
    um roteiro que voce ja mudou. O que pega e o `input_hash` do storyboard,
    que carrega o digest do roteiro.
    """
    config_path, work, provider = projeto
    cli(config_path, "approve", "script", "esteira-de-video")
    cli(config_path, "approve", "storyboard", "esteira-de-video")

    (work / "script.md").write_text(
        ROTEIRO.replace("paciencia infinita", "paciencia que nao acaba"),
        encoding="utf-8")

    with pytest.raises(SystemExit, match="outra versao do roteiro"):
        cli(config_path, "images", "esteira-de-video")
    assert provider.calls == []


def test_do_roteiro_ao_video_montado(projeto, gravacao, capsys):
    """O caminho inteiro, comando por comando, como ele vai ser usado."""
    config_path, work, provider = projeto

    assert cli(config_path, "approve", "script", "esteira-de-video") == 0
    assert cli(config_path, "approve", "storyboard", "esteira-de-video") == 0

    # --- estagio 3: as imagens, antes de gravar -------------------------
    assert cli(config_path, "images", "esteira-de-video", "--dry-run") == 0
    assert provider.calls == [], "o dry-run gastou"
    assert not (work / "assets.json").exists()

    assert cli(config_path, "images", "esteira-de-video") == 0
    assets = read_json(work / "assets.json", Assets)
    assert len(assets.items) == 3
    assert len(provider.calls) == 3
    assert sorted(i.beat_id for i in assets.items) == [1, 2, 3]

    assert cli(config_path, "approve", "images", "esteira-de-video") == 0

    # --- planta a transcricao, para nao baixar o whisper ----------------
    #     O `shoot` roda o ingest primeiro; o transcript e cacheado pelo
    #     hash do AUDIO, que so existe depois disso.
    from pipeline.config import load_config
    from pipeline.stages import ingest

    cfg = load_config(config_path)
    manifest = ingest.run(gravacao, cfg, slug="esteira-de-video")
    write_json(work / "transcript.json", um_transcript(manifest.audio_hash))

    # --- depois de gravar: um comando ----------------------------------
    assert cli(config_path, "shoot", "esteira-de-video", str(gravacao)) == 0

    saida = work / "final.mp4"
    assert saida.exists()
    duracao = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(saida)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert float(duracao) == pytest.approx(DURACAO, abs=0.3)

    # --- e os artefatos do alinhamento ---------------------------------
    alinhado = read_json(work / "assets.aligned.json", Assets)
    relatorio = read_json(work / "align.json", Placements)

    assert relatorio.n_orphans == 0
    assert all(c.method == "literal" for c in relatorio.placements), \
        "o embedding foi usado num roteiro seguido a risca"
    assert all(i.segment_index >= 0 for i in alinhado.items)
    # o assets.json aprovado continua indexado por beat
    assert all(i.segment_index == -1 for i in read_json(work / "assets.json", Assets).items)


def test_a_gravacao_entra_no_work_do_roteiro(projeto, gravacao):
    """O slug vem da IDEIA, meses antes de existir arquivo. Sem o slug
    explicito o ingest abriria um `work/` novo e o align nao acharia nada."""
    config_path, work, provider = projeto
    from pipeline.config import load_config
    from pipeline.stages import ingest

    cfg = load_config(config_path)
    manifest = ingest.run(gravacao, cfg, slug="esteira-de-video")

    assert manifest.slug == "esteira-de-video"
    assert (work / "manifest.json").exists()
    assert (work / "audio.wav").exists()
    # e nenhum diretorio novo derivado do nome do arquivo
    assert sorted(p.name for p in (cfg.work_dir).iterdir()) == ["esteira-de-video"]


def test_status_mostra_os_tres_portoes(projeto, capsys):
    config_path, work, provider = projeto
    cli(config_path, "approve", "script", "esteira-de-video")

    assert cli(config_path, "status") == 0
    saida = capsys.readouterr().out

    assert "roteiro" in saida and "storyboard" in saida and "imagens" in saida
    assert "aprovado" in saida


def test_align_isolado_depois_do_shoot(projeto, gravacao):
    """Rodar o estagio sozinho tem que funcionar: e como voce recalibra o
    alinhamento sem refazer a transcricao nem pagar imagem de novo."""
    config_path, work, provider = projeto
    from pipeline.config import load_config
    from pipeline.stages import ingest

    cli(config_path, "approve", "script", "esteira-de-video")
    cli(config_path, "approve", "storyboard", "esteira-de-video")
    cli(config_path, "images", "esteira-de-video")
    cli(config_path, "approve", "images", "esteira-de-video")

    cfg = load_config(config_path)
    manifest = ingest.run(gravacao, cfg, slug="esteira-de-video")
    write_json(work / "transcript.json", um_transcript(manifest.audio_hash))
    cli(config_path, "trim", "esteira-de-video")

    assert cli(config_path, "align", "esteira-de-video") == 0
    assert (work / "edl.json").exists()
    assert len(provider.calls) == 3, "o align pagou imagem"


def test_o_portao_das_imagens_para_o_align(projeto, gravacao):
    config_path, work, provider = projeto
    from pipeline.config import load_config
    from pipeline.stages import ingest

    cli(config_path, "approve", "script", "esteira-de-video")
    cli(config_path, "approve", "storyboard", "esteira-de-video")
    cli(config_path, "images", "esteira-de-video")
    # de proposito: NAO aprova as imagens

    cfg = load_config(config_path)
    manifest = ingest.run(gravacao, cfg, slug="esteira-de-video")
    write_json(work / "transcript.json", um_transcript(manifest.audio_hash))
    cli(config_path, "trim", "esteira-de-video")

    with pytest.raises(SystemExit, match="pipeline approve images"):
        cli(config_path, "align", "esteira-de-video")
    assert not (work / "edl.json").exists()


def test_editar_o_roteiro_depois_de_tudo_aprovado_revoga(projeto):
    """A aprovacao guarda o digest, entao mexer no arquivo derruba o portao —
    e o estagio seguinte para em vez de rodar com o aval da versao antiga."""
    config_path, work, provider = projeto
    cli(config_path, "approve", "script", "esteira-de-video")
    cli(config_path, "approve", "storyboard", "esteira-de-video")

    (work / "script.md").write_text(
        ROTEIRO.replace("paciencia infinita", "paciencia sem fim"), encoding="utf-8")

    from pipeline import progress
    assert progress.script_gate(work).state == progress.EDITED
    assert approval.read(work, "storyboard") is not None
