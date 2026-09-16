"""A interface de revisao e aprovacao. FastAPI + HTML renderizado no servidor.

Tres decisoes moldam este modulo:

**A UI nao e dona de estado nenhum.** Toda pagina le o disco na hora, e todo
POST escreve no disco e redireciona. Nao existe sessao, nem cache em memoria,
nem estado que morre quando o servidor cai. Fechar o navegador no meio nao
perde nada, e a CLI continua fazendo tudo sozinha — as duas olham os mesmos
arquivos em `work/<slug>/`. E o que ja valia para os estagios: o portao de
aprovacao e um arquivo (ver `approval.py`).

**Nenhum GET gasta dinheiro.** Abrir uma pagina le artefato pronto e nada
mais. A tela do storyboard nao chama o agente: se `storyboard.json` nao
existe, ela diz qual comando rodar. Uma pagina que dispara uma chamada de
Opus ao ser recarregada seria uma armadilha — e navegador recarrega por conta
propria.

**Escrever o roteiro valida antes de gravar.** O texto vem de um textarea, e
salvar um Markdown quebrado por cima do arquivo bom destruiria o unico
artefato que voce escreveu a mao. Entao o texto e parseado primeiro e so
gravado se passar; se nao passar, a pagina volta com o erro E com o seu texto
intacto no campo, para voce consertar em vez de redigitar.

Sem Jinja de proposito: sao tres paginas pequenas e uma dependencia a menos.
Todo texto de fora — o seu roteiro, o que o modelo escreveu — passa por
`escape`, senao um `<` legitimo no roteiro quebraria a pagina.
"""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse

from . import approval, progress
from .config import Config
from .schemas import Script, Storyboard, StoryboardBeat
from .script import ScriptFormatError, parse as parse_script, spoken_seconds, words
from .util import read_json

STYLE = """
:root {
  --fundo: #14181b; --painel: #1b2227; --linha: #2b353c;
  --texto: #dfe6ea; --fraco: #8b9aa3; --claro: #f4f8fa;
  --ok: #6fca8f; --atencao: #e8b04b; --erro: #e8726b;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1rem; background: var(--fundo); color: var(--texto);
  font: 16px/1.6 ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
}
main { max-width: 60rem; margin: 0 auto; }
a { color: var(--claro); }
h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
h1 a { text-decoration: none; color: var(--fraco); font-weight: 400; }
p.sub { color: var(--fraco); margin: 0 0 2rem; }
section {
  background: var(--painel); border: 1px solid var(--linha);
  border-radius: 8px; padding: 1.25rem; margin-bottom: 1.25rem;
}
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: .5rem .5rem; border-bottom: 1px solid var(--linha); }
th { color: var(--fraco); font-weight: 500; font-size: .85rem; }
tr:last-child td { border-bottom: none; }
.tag { font-size: .8rem; padding: .1rem .5rem; border-radius: 99px; border: 1px solid var(--linha); }
.aprovado { color: var(--ok); border-color: var(--ok); }
.editado { color: var(--atencao); border-color: var(--atencao); }
.ausente { color: var(--fraco); }
.erro { color: var(--erro); }
textarea {
  width: 100%; min-height: 26rem; background: var(--fundo); color: var(--texto);
  border: 1px solid var(--linha); border-radius: 6px; padding: .75rem;
  font: 14px/1.7 ui-monospace, SFMono-Regular, Menlo, monospace;
}
button {
  background: var(--claro); color: #12171a; border: 0; border-radius: 6px;
  padding: .55rem 1.1rem; font-size: .95rem; font-weight: 600; cursor: pointer;
}
button.leve { background: transparent; color: var(--texto); border: 1px solid var(--linha); }
.beat { border-top: 1px solid var(--linha); padding: 1rem 0; }
.beat:first-of-type { border-top: 0; padding-top: 0; }
.beat h3 { margin: 0 0 .5rem; font-size: 1rem; }
.beat h3 span { color: var(--fraco); font-weight: 400; font-size: .85rem; }
.grade { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }
.grade dt { color: var(--fraco); font-size: .8rem; margin-top: .6rem; }
.grade dd { margin: 0; }
.fala { color: var(--fraco); font-size: .95rem; }
mark { background: #3c4a2a; color: var(--claro); padding: 0 .15rem; border-radius: 3px; }
.roteiro { white-space: pre-wrap; font: 14px/1.7 ui-monospace, Menlo, monospace; }
code { background: var(--fundo); padding: .1rem .35rem; border-radius: 4px; font-size: .9em; }
@media (max-width: 44rem) { .grade { grid-template-columns: 1fr; } }
"""

CLASSES = {
    progress.APPROVED: "aprovado",
    progress.EDITED: "editado",
    progress.ABSENT: "ausente",
    progress.WRITTEN: "",
}


class NotFound(Exception):
    """Slug que nao existe em `work/`. Vira uma pagina com a lista, nao um 500."""

    def __init__(self, slug: str) -> None:
        self.slug = slug


def page(title: str, body: str, *, back: bool = True, status: int = 200) -> HTMLResponse:
    voltar = ' <a href="/">&larr; todos</a>' if back else ""
    return HTMLResponse(
        status_code=status,
        content=(
            "<!doctype html>\n"
            f'<html lang="pt-BR"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{escape(title)}</title><style>{STYLE}</style></head>"
            f"<body><main><h1>{escape(title)}{voltar}</h1>{body}</main></body></html>"
        ),
    )


def badge(gate: progress.Gate) -> str:
    classe = CLASSES[gate.state]
    return f'<span class="tag {classe}">{escape(gate.state)}</span>'


def anchor_markup(beat_text: str, anchor: str) -> str:
    """O texto do beat com o ancora destacado, ou um aviso se ele nao esta la.

    O destaque e o que faz a revisao valer: ele mostra em que ponto da FALA
    aquela imagem entra, que e a unica pergunta que essa tela existe para
    responder.

    Casa por texto cru com fronteira de palavra, e nao pela posicao em
    palavras normalizadas que o `storyboard` guarda: a normalizacao troca
    pontuacao por espaco, entao um token cru pode virar dois e os indices
    deixam de corresponder. Destaque deslocado seria pior que destaque
    nenhum — plausivel demais para alguem notar numa revisao rapida, que e
    exatamente o erro que esta arquitetura existe para nao cometer. Quando
    nao casa literalmente, a tela DIZ isso, porque e informacao de revisao:
    quer dizer que o modelo parafraseou o proprio ancora.
    """
    achado = re.search(rf"(?<!\w){re.escape(anchor.strip())}(?!\w)", beat_text)
    if achado is None:
        return (
            f'<p class="fala">{escape(beat_text)}</p>'
            f'<p class="erro">o ancora nao aparece literal neste texto — '
            f"o alinhamento vai cair na busca por similaridade</p>"
        )
    inicio, fim = achado.span()
    return (
        f'<p class="fala">{escape(beat_text[:inicio])}'
        f"<mark>{escape(beat_text[inicio:fim])}</mark>"
        f'{escape(beat_text[fim:])}</p>'
    )


def beat_card(beat: StoryboardBeat, script: Script) -> str:
    do_roteiro = script.beat(beat.beat_id)
    titulo = f" — {do_roteiro.title}" if do_roteiro and do_roteiro.title else ""
    planos = f"{beat.sub_shots} plano{'s' if beat.sub_shots != 1 else ''}"
    texto = do_roteiro.text if do_roteiro else ""
    faltando = "" if do_roteiro else (
        '<p class="erro">este beat nao existe mais no roteiro — '
        "o storyboard precisa ser refeito</p>"
    )
    return (
        f'<div class="beat"><h3>beat {beat.beat_id}{escape(titulo)} '
        f"<span>{planos} de {beat.seconds_per_shot:g}s = "
        f"{beat.screen_seconds:g}s na tela</span></h3>"
        f'<div class="grade"><div><dl>'
        f"<dt>imagem</dt><dd>{escape(beat.concept)}</dd>"
        f"<dt>tags</dt><dd>{escape(', '.join(beat.concept_tags))}</dd>"
        f"<dt>porque</dt><dd>{escape(beat.rationale)}</dd>"
        f"</dl></div><div><dl><dt>entra em</dt><dd>"
        f"{faltando or anchor_markup(texto, beat.script_anchor)}"
        f"</dd></dl></div></div></div>"
    )


def work_for(config: Config, slug: str) -> Path:
    """O diretorio daquele slug, ou `NotFound`.

    A validacao e por PERTENCER a lista de diretorios que existem, e nao por
    sanitizar a string: e o unico jeito que nao depende de eu ter pensado em
    todas as formas de escrever `..`. A UI escreve arquivo (o roteiro), entao
    a diferenca importa mesmo servindo em localhost.

    Mora no modulo e nao dentro de `create_app` para poder ser testada
    direto. Pelo HTTP ela quase nao da para exercitar: o cliente normaliza
    `../` antes de mandar, entao um teste que so faz `GET /../x` passa sem
    tocar nesta funcao — passa pelo motivo errado, e foi o que aconteceu na
    primeira versao deste codigo.
    """
    for candidato in config.work_dir.glob("*"):
        if candidato.is_dir() and candidato.name == slug:
            return candidato
    raise NotFound(slug)


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="youtube-auto", docs_url=None, redoc_url=None)

    @app.exception_handler(NotFound)
    def _nao_achou(_request, exc: NotFound):
        disponiveis = "".join(
            f'<li><a href="/{escape(s.slug)}/script">{escape(s.slug)}</a></li>'
            for s in progress.all_states(config)
        )
        return page(
            "nao encontrado",
            f"<section><p>Nao existe <code>{escape(exc.slug)}</code> em "
            f"<code>{escape(str(config.work_dir))}</code>.</p>"
            f"<ul>{disponiveis or '<li>(nenhum video ainda)</li>'}</ul></section>",
            status=404,
        )

    # ---------------------------------------------------------------- indice

    @app.get("/", response_class=HTMLResponse)
    def index():
        estados = progress.all_states(config)
        if not estados:
            return page(
                "youtube-auto",
                "<section><p>Nenhum video em <code>work/</code> ainda.</p>"
                "<p>Comece pela ideia:</p>"
                "<p><code>pipeline script \"sua ideia aqui\"</code></p></section>",
                back=False,
            )

        linhas = ""
        for estado in estados:
            erro = estado.script.error or estado.storyboard.error
            aviso = f'<br><span class="erro">{escape(erro)}</span>' if erro else ""
            linhas += (
                f"<tr><td><a href=\"/{escape(estado.slug)}/script\">"
                f"{escape(estado.slug)}</a>{aviso}</td>"
                f"<td>{badge(estado.script)}</td>"
                f"<td>{badge(estado.storyboard)}</td>"
                f"<td>{'sim' if estado.recorded else '-'}</td>"
                f'<td class="ausente">{escape(estado.stage)}</td></tr>'
            )
        return page(
            "youtube-auto",
            "<section><table><tr><th>video</th><th>roteiro</th>"
            "<th>storyboard</th><th>gravado</th><th>estado</th></tr>"
            f"{linhas}</table></section>"
            '<section><p class="ausente">"editado!" quer dizer que o arquivo '
            "mudou depois de aprovado: a aprovacao valia para outra versao e "
            "precisa ser dada de novo.</p></section>",
            back=False,
        )

    # --------------------------------------------------------------- roteiro

    def script_page(work: Path, *, texto: str | None = None, erro: str = "") -> HTMLResponse:
        """A pagina do roteiro.

        `texto` vindo preenchido e o caminho do POST que nao passou na
        validacao: mostra o que a pessoa escreveu, nao o que esta no disco.
        """
        slug = work.name
        caminho = work / "script.md"
        cru = texto if texto is not None else (
            caminho.read_text(encoding="utf-8") if caminho.exists() else ""
        )

        cabecalho = ""
        beats = ""
        gate = progress.script_gate(work)
        if erro:
            cabecalho = f'<section><p class="erro">{escape(erro)}</p></section>'
        elif not caminho.exists():
            cabecalho = (
                "<section><p>Este video ainda nao tem roteiro.</p>"
                f"<p><code>pipeline script \"sua ideia\" --slug {escape(slug)}</code>"
                "</p></section>"
            )
        elif gate.error:
            cabecalho = f'<section><p class="erro">{escape(gate.error)}</p></section>'
        else:
            script = parse_script(cru)
            falado = spoken_seconds(script, config.script.words_per_minute)
            alvo = script.duration_target_seconds
            comparacao = f" (alvo {alvo:.0f}s)" if alvo else ""
            cabecalho = (
                f"<section><dl class=\"grade\">"
                f"<div><dt>assunto</dt><dd>{escape(script.subject)}</dd>"
                f"<dt>argumento</dt><dd>{escape(script.argument)}</dd></div>"
                f"<div><dt>o espectador leva</dt><dd>{escape(script.viewer_takeaway)}</dd>"
                f"<dt>tamanho</dt><dd>{len(script.beats)} beats, {words(script)} "
                f"palavras, ~{falado:.0f}s falados{comparacao}</dd></div></dl>"
                f"<p>{badge(gate)} "
                f'<a href="/{escape(slug)}/storyboard">storyboard &rarr;</a></p>'
                f"{_approve_form(slug, 'script', gate)}</section>"
            )
            beats = "<section>" + "".join(
                f'<div class="beat"><h3>beat {b.id}'
                f"{escape(' — ' + b.title) if b.title else ''}</h3>"
                f'<p class="roteiro">{escape(b.text)}</p></div>'
                for b in script.beats
            ) + "</section>"

        editor = (
            f'<section><form method="post" action="/{escape(slug)}/script">'
            f'<textarea name="texto" spellcheck="false">{escape(cru)}</textarea>'
            f'<p><button type="submit">Salvar</button> '
            f'<span class="ausente">salva em '
            f"<code>{escape(str(caminho))}</code> depois de validar o formato"
            f"</span></p></form></section>"
        )
        return page(f"roteiro · {slug}", cabecalho + editor + beats)

    def _approve_form(slug: str, gate_name: str, gate: progress.Gate) -> str:
        if not gate.pending:
            return ""
        rotulo = "Aprovar" if gate.state == progress.WRITTEN else "Aprovar de novo"
        return (
            f'<form method="post" action="/{escape(slug)}/{gate_name}/approve">'
            f'<button type="submit">{rotulo}</button></form>'
        )

    @app.get("/{slug}/script", response_class=HTMLResponse)
    def get_script(slug: str):
        return script_page(work_for(config, slug))

    @app.post("/{slug}/script", response_class=HTMLResponse)
    def post_script(slug: str, texto: str = Form("")):
        work = work_for(config, slug)
        # O textarea devolve CRLF. Sem normalizar, o `\r` sobrevive DENTRO do
        # texto do beat (o parser junta linhas com "\n" e so tira espaco das
        # pontas) e entra no prompt e no digest.
        limpo = texto.replace("\r\n", "\n").replace("\r", "\n")
        try:
            parse_script(limpo)
        except ScriptFormatError as exc:
            return script_page(work, texto=limpo, erro=str(exc))

        # Grava VERBATIM, nao o `render()` do que foi parseado: o render
        # normaliza e descartaria comentario antes do primeiro beat, ordem do
        # frontmatter e espacamento. Parsear garante que o proximo estagio le;
        # nao autoriza reescrever o que voce digitou.
        (work / "script.md").write_text(limpo, encoding="utf-8")
        return RedirectResponse(f"/{slug}/script", status_code=303)

    @app.post("/{slug}/script/approve")
    def approve_script(slug: str):
        work = work_for(config, slug)
        script = parse_script((work / "script.md").read_text(encoding="utf-8"))
        approval.grant(work, "script", script.digest())
        return RedirectResponse(f"/{slug}/script", status_code=303)

    # ------------------------------------------------------------ storyboard

    @app.get("/{slug}/storyboard", response_class=HTMLResponse)
    def get_storyboard(slug: str):
        work = work_for(config, slug)
        gate = progress.storyboard_gate(work)
        titulo = f"storyboard · {slug}"

        if gate.state == progress.ABSENT:
            recado = gate.error or "Este video ainda nao tem storyboard."
            classe = "erro" if gate.error else ""
            return page(titulo,
                        f'<section><p class="{classe}">{escape(recado)}</p>'
                        f"<p><code>pipeline storyboard {escape(slug)}</code></p>"
                        f'<p class="ausente">Esta pagina nao roda o agente: '
                        f"recarregar o navegador nao pode gastar uma chamada de "
                        f"Opus.</p></section>")

        board = read_json(work / "storyboard.json", Storyboard)
        script = parse_script((work / "script.md").read_text(encoding="utf-8"))
        planos = sum(b.sub_shots for b in board.beats)
        return page(
            titulo,
            f"<section><p>{board.n_images} "
            f"{'imagens' if board.n_images != 1 else 'imagem'}, {planos} planos, "
            f"{board.total_screen_seconds:.0f}s na tela &nbsp; {badge(gate)} "
            f'&nbsp; <a href="/{escape(slug)}/script">&larr; roteiro</a></p>'
            f"{_approve_form(slug, 'storyboard', gate)}</section>"
            "<section>"
            + "".join(beat_card(b, script) for b in board.beats)
            + "</section>",
        )

    @app.post("/{slug}/storyboard/approve")
    def approve_storyboard(slug: str):
        """Aprova o `input_hash` que esta em disco.

        Nao recalcula o storyboard para descobrir o hash: aprovar o que esta
        na tela e o que a pessoa acabou de ler, e recalcular podia gastar uma
        chamada. Aprovar um hash velho tambem nao deixa nada passar — o
        estagio seguinte compara a propria chave de cache de novo, e um
        storyboard desatualizado e recalculado e volta a precisar de
        aprovacao.
        """
        work = work_for(config, slug)
        board = read_json(work / "storyboard.json", Storyboard)
        approval.grant(work, "storyboard", board.input_hash)
        return RedirectResponse(f"/{slug}/storyboard", status_code=303)

    return app


def serve(config: Config, *, host: str, port: int) -> None:
    import uvicorn

    log_url = f"http://{host}:{port}"
    # flush: com a saida num pipe, sem isso a URL so apareceria quando o
    # servidor morresse — e ele nao morre.
    print(f"\n  {log_url}\n  work/ em {config.work_dir}\n  ctrl-c para parar\n", flush=True)
    uvicorn.run(create_app(config), host=host, port=port, log_level="warning")
