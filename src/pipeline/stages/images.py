"""Estagio 3: o storyboard aprovado vira imagens. ANTES de a camera ligar.

E a mesma resolucao do estagio 4 do caminho gravacao-primeiro — banco, depois
stock, depois geracao, com o mesmo teto de orcamento — pedida por BEAT em vez
de por segmento de EDL. Ver `resolver.ImageRequest`: o resolver nunca precisou
de uma EDL, e e por isso que um resolver so serve aos dois caminhos.

A diferenca que importa e o que ainda nao existe aqui: nao ha gravacao, nao ha
transcript, nao ha EDL. A imagem esta amarrada a um beat do roteiro, e onde
aquele beat cai no tempo so o `align` descobre depois de voce gravar. Por isso
o `AssetItem` sai com `beat_id` e `segment_index = -1`.

Uma imagem por beat, nao por sub-plano: os sub-planos saem todos da mesma
imagem, recortando regioes diferentes dela. E o que faz uma faixa de 10s de
b-roll custar uma imagem em vez de quatro.
"""

from __future__ import annotations

from .. import approval
from ..config import Config
from ..log import log, stage
from ..resolver import BudgetExceeded, from_storyboard
from ..schemas import Assets, Script, Storyboard
from ..util import read_json_if_fresh, text_hash, write_json
from .assets import build_resolver, open_bank, print_estimate, provider_key
from .storyboard import cache_key as storyboard_key


def cache_key(board: Storyboard, config: Config) -> str:
    """Identidade das imagens: o storyboard mais o que decide a imagem.

    O `input_hash` do storyboard ja carrega o roteiro, as regras editoriais, a
    duracao do sub-plano e o estilo (ver `stages/storyboard.cache_key`), entao
    aqui basta ele — e `provider_key`, que e o que muda de imagem sem mudar de
    conceito.
    """
    return text_hash(board.input_hash, *provider_key(config))


def run(
    script: Script,
    board: Storyboard,
    config: Config,
    *,
    dry_run: bool = False,
) -> Assets:
    work = config.work_dir / script.slug
    assets_path = work / ("assets.dryrun.json" if dry_run else "assets.json")
    key = cache_key(board, config)

    # Duas conferencias antes de gastar, e a segunda nao e redundante.
    #
    # O portao diz que voce aprovou ESTE storyboard. Mas o portao do roteiro
    # nao e conferido aqui e nao deveria bastar: aprovando o roteiro, gerando
    # o storyboard, aprovando o storyboard e DEPOIS editando o roteiro, os
    # dois portoes continuam de pe — e as imagens ilustrariam um roteiro que
    # voce ja mudou. O que fecha isso e comparar o `input_hash` do storyboard
    # com a chave que ele teria hoje: ela inclui o digest do roteiro e o que
    # mais decide o storyboard, entao um storyboard velho e detectado em vez
    # de virar dinheiro gasto no desenho errado.
    approval.require(
        work, "storyboard", board.input_hash,
        what="O storyboard",
        command=f"pipeline approve storyboard {script.slug}",
    )
    atual = storyboard_key(script, config)
    if board.input_hash != atual:
        raise approval.NotApproved(
            "O storyboard em disco foi feito para outra versao do roteiro (ou "
            "com outro config).\n"
            f"Refaca e aprove de novo:  pipeline storyboard {script.slug}"
        )

    cached = read_json_if_fresh(assets_path, Assets, key)
    if cached is not None and cached.dry_run == dry_run:
        log("images.cached", slug=script.slug, cost=f"${cached.total_cost_usd:.2f}")
        return cached

    with stage("images", slug=script.slug, beats=board.n_images, dry_run=dry_run):
        bank = open_bank(config)
        try:
            if not dry_run:
                # Antes de baixar ou gerar qualquer coisa: se o modelo de
                # embedding nao carrega, melhor saber agora que depois de pagar.
                bank.warmup()
            resolver = build_resolver(config, bank, dry_run=dry_run)
            try:
                assets = resolver.resolve(
                    from_storyboard(board.beats), dry_run=dry_run)
            except BudgetExceeded as exc:
                print_estimate(exc.estimate, config)
                raise
            assets.input_hash = key
            write_json(assets_path, assets)
            log("images.ok", cost=f"${assets.total_cost_usd:.4f}", **assets.by_origin)
            return assets
        finally:
            bank.close()


def regenerate(
    script: Script,
    board: Storyboard,
    assets: Assets,
    beat_id: int,
    config: Config,
) -> Assets:
    """Troca a imagem de UM beat por uma nova. Gasta.

    Tres coisas acontecem, e as tres importam:

    O provider e chamado a forca, sem passar por banco nem stock — ver
    `AssetResolver.regenerate` para por que uma resolucao normal devolveria a
    imagem recusada de volta.

    O asset antigo e APAGADO do banco. Deixar a linha la faz o proximo video
    com um conceito parecido reusar exatamente a imagem que voce rejeitou, e em
    silencio, porque reuso do banco nao passa por aprovacao.

    E o `assets.json` e reescrito, o que muda o `digest()` e portanto REVOGA a
    aprovacao das imagens. E o comportamento certo: voce aprovou um conjunto,
    e este e outro. Ver `Assets.digest`.
    """
    work = config.work_dir / script.slug
    beat = next((b for b in board.beats if b.beat_id == beat_id), None)
    if beat is None:
        raise ValueError(f"o storyboard nao tem beat {beat_id}")

    antigo = next((i for i in assets.items if i.beat_id == beat_id), None)
    if antigo is None:
        raise ValueError(f"nao ha imagem para o beat {beat_id} em assets.json")

    unit = config.image_provider.replicate.cost_usd_per_image
    teto = config.budget.max_usd_per_video
    if assets.total_cost_usd + unit > teto:
        raise BudgetExceeded(assets.estimate.model_copy(update={
            "worst_case_usd": round(assets.total_cost_usd + unit, 4),
            "budget_usd": teto,
            "within_budget": False,
        }))

    with stage("regenerate", slug=script.slug, beat=beat_id):
        bank = open_bank(config)
        try:
            resolver = build_resolver(config, bank, dry_run=False)
            novo = resolver.regenerate(from_storyboard([beat])[0])
            if antigo.asset_id is not None:
                bank.forget(antigo.asset_id)
        finally:
            bank.close()

    items = [novo if i.beat_id == beat_id else i for i in assets.items]
    by_origin: dict[str, int] = {}
    for item in items:
        by_origin[item.origin] = by_origin.get(item.origin, 0) + 1

    atualizado = assets.model_copy(update={
        "items": items,
        "total_cost_usd": round(sum(i.cost_usd for i in items), 4),
        "by_origin": by_origin,
    })
    write_json(work / "assets.json", atualizado)
    log("regenerate.ok", beat=beat_id, file=novo.path,
        custo=f"${novo.cost_usd:.4f}", detail="a aprovacao das imagens foi revogada")
    return atualizado


def render_text(assets: Assets, board: Storyboard, config: Config) -> str:
    """A revisao das imagens no terminal, antes de aprovar.

    Mostra o conceito ao lado do arquivo, porque a pergunta desta revisao e se
    a imagem que saiu e a imagem que foi pedida — e o caminho do arquivo
    sozinho nao responde isso.
    """
    por_beat = {b.beat_id: b for b in board.beats}
    linhas = ["", "  Imagens", "  " + "-" * 72]
    for item in assets.items:
        beat = por_beat.get(item.beat_id)
        planos = f"{beat.sub_shots} plano{'s' if beat and beat.sub_shots != 1 else ''}" if beat else "?"
        linhas.append(f"  beat {item.beat_id}   [{item.origin}]   {planos}")
        if beat is not None:
            linhas.append(f"    conceito: {beat.concept}")
        linhas.append(f"    arquivo:  {item.path or '(cor solida)'}")
        if item.note:
            linhas.append(f"    nota:     {item.note}")
        linhas.append("")
    rate = config.budget.brl_per_usd
    linhas += [
        "  " + "-" * 72,
        f"  {len(assets.items)} imagens, "
        f"USD {assets.total_cost_usd:.4f}   BRL {assets.total_cost_usd * rate:.2f}",
        "",
    ]
    return "\n".join(linhas)
