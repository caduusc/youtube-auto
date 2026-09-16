"""O roteiro, em Markdown com frontmatter.

Markdown e nao JSON de proposito: o roteiro e o unico artefato do pipeline que
existe para ser LIDO E EDITADO a mao. JSON com texto longo dentro de string
escapada e hostil para revisar, e revisar e o ponto deste estagio.

O formato:

    ---
    slug: ...
    subject: ...
    ---

    ## beat 1 — abertura
    Texto do roteiro.

    ## beat 2 — o problema
    Mais texto.

O `id` do beat vem do numero no cabecalho, nao da ordem no arquivo. Isso e o
que permite voce reordenar ou apagar um beat sem renumerar tudo — o storyboard
referencia por id, e um id que desaparece e detectado em vez de virar
silenciosamente outro beat.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .schemas import Script, ScriptBeat

# `## beat 3 — titulo` ou `## beat 3 - titulo` ou so `## beat 3`. O travessao
# aceita as duas formas porque quem edita a mao digita o que o teclado tem.
BEAT_HEADING = re.compile(r"^##\s+beat\s+(\d+)\s*(?:[—–-]\s*(.*))?$", re.IGNORECASE)
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

REQUIRED_FIELDS = ("slug", "subject", "argument", "viewer_takeaway")


class ScriptFormatError(ValueError):
    """O roteiro no disco nao casa com o formato.

    Mensagem sempre aponta o que fazer: este arquivo e editado a mao, entao o
    erro e um recado para uma pessoa, nao um traceback.
    """


def parse(text: str) -> Script:
    match = FRONTMATTER.match(text)
    if match is None:
        raise ScriptFormatError(
            "o roteiro precisa comecar com um bloco de frontmatter delimitado por "
            "'---' na primeira linha e '---' de novo depois dos campos"
        )

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ScriptFormatError(f"o frontmatter nao e YAML valido: {exc}") from exc
    if not isinstance(meta, dict):
        raise ScriptFormatError("o frontmatter precisa ser um mapa de campo: valor")

    faltando = [f for f in REQUIRED_FIELDS if not str(meta.get(f, "")).strip()]
    if faltando:
        raise ScriptFormatError(
            f"faltam campos no frontmatter: {', '.join(faltando)}"
        )

    beats = parse_beats(text[match.end():])
    if not beats:
        raise ScriptFormatError(
            "o roteiro nao tem beat nenhum. Cada trecho comeca com um cabecalho "
            "'## beat N — titulo'"
        )

    return Script(
        slug=str(meta["slug"]).strip(),
        subject=str(meta["subject"]).strip(),
        argument=str(meta["argument"]).strip(),
        viewer_takeaway=str(meta["viewer_takeaway"]).strip(),
        duration_target_seconds=float(meta.get("duration_target_seconds") or 0.0),
        beats=beats,
    )


def parse_beats(body: str) -> list[ScriptBeat]:
    """Os beats do corpo, na ordem em que aparecem.

    Texto antes do primeiro cabecalho e ignorado de proposito: e onde uma nota
    para voce mesmo cabe sem virar conteudo do video.
    """
    beats: list[ScriptBeat] = []
    atual: dict | None = None
    linhas: list[str] = []
    vistos: set[int] = set()

    def fechar() -> None:
        if atual is None:
            return
        texto = "\n".join(linhas).strip()
        if not texto:
            raise ScriptFormatError(
                f"o beat {atual['id']} esta vazio; escreva o texto ou apague o cabecalho"
            )
        beats.append(ScriptBeat(id=atual["id"], title=atual["title"], text=texto))

    for linha in body.splitlines():
        cabecalho = BEAT_HEADING.match(linha.strip())
        if cabecalho is None:
            if atual is not None:
                linhas.append(linha)
            continue

        fechar()
        numero = int(cabecalho.group(1))
        if numero in vistos:
            raise ScriptFormatError(
                f"o beat {numero} aparece duas vezes; os ids tem que ser unicos "
                "porque o storyboard referencia por id"
            )
        vistos.add(numero)
        atual = {"id": numero, "title": (cabecalho.group(2) or "").strip()}
        linhas = []

    fechar()
    return beats


def render(script: Script) -> str:
    """De volta para Markdown. `parse(render(s)) == s` — ver os testes.

    O round-trip importa porque o estagio 1 escreve o arquivo, VOCE edita, e o
    estagio 2 le de novo. Se a ida e a volta nao fossem iguais, uma edicao sua
    poderia ser silenciosamente desfeita.
    """
    meta = {
        "slug": script.slug,
        "subject": script.subject,
        "argument": script.argument,
        "viewer_takeaway": script.viewer_takeaway,
        "duration_target_seconds": script.duration_target_seconds,
    }
    frente = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=10**6)
    partes = ["---", frente.rstrip("\n"), "---", ""]
    for beat in script.beats:
        titulo = f"## beat {beat.id}" + (f" — {beat.title}" if beat.title else "")
        partes += [titulo, "", beat.text, ""]
    return "\n".join(partes).rstrip("\n") + "\n"


def read(path: Path) -> Script:
    try:
        texto = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ScriptFormatError(f"roteiro nao encontrado: {path}") from None
    try:
        return parse(texto)
    except ScriptFormatError as exc:
        raise ScriptFormatError(f"{path}: {exc}") from None


def write(path: Path, script: Script) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(script), encoding="utf-8")


def words(script: Script) -> int:
    return sum(len(beat.text.split()) for beat in script.beats)


def spoken_seconds(script: Script, words_per_minute: float) -> float:
    """Quanto tempo o roteiro deve durar falado.

    Serve para o estagio 1 conferir se acertou a duracao pedida, e para o
    storyboard saber quanto tempo cada beat ocupa antes de existir gravacao.
    """
    return words(script) / max(words_per_minute, 1.0) * 60.0
