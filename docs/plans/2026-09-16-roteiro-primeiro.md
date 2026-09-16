# Roteiro primeiro: agentes, aprovação e sub-planos

Plano de desenho. Nada implementado ainda.

## O que muda, e por quê

O pipeline hoje é **descritivo**: recebe uma gravação que já existe e tenta inferir
intenção visual de fala improvisada. É por isso que "as imagens não pegaram o
contexto" — o modelo adivinha o que o vídeo quer mostrar a partir da transcrição.

O novo é **prescritivo**: o roteiro existe e é aprovado antes de a câmera ligar, e
as imagens saem dele. A intenção deixa de ser inferida.

## O problema de "delírio" não é o modelo de imagem

Levantado antes de decidir provider, porque a conclusão inverte a intuição.

`flux-schnell` é destilado em **4 passos**. A documentação do próprio modelo:
"good basic prompt following but **may simplify or miss elements in complex
prompts**". `flux-dev` (20-30 passos): "**strong prompt adherence**".

E o prompt que mandamos hoje tem ~65 palavras, das quais **~45 são estilo**:

    <concept: 20 palavras>, <style_suffix: 45 palavras>

Num modelo de 4 passos, o que é descartado primeiro tende a ser o conteúdo —
palavras de estilo são mais "prompt-shaped". Ou seja: pagamos o prompt inteiro e
o modelo lê principalmente o estilo.

Três consertos, em ordem de alavancagem:

1. **`flux-dev` em vez de `flux-schnell`.** 8.3x o custo ($0.003 -> $0.025), e é o
   único que ataca aderência diretamente.
2. **Encurtar o `style_suffix`.** De ~45 para ~20 palavras dobra a fração do prompt
   que descreve o conteúdo. Grátis.
3. **Tirar as negações do prompt positivo.** Hoje o sufixo carrega cinco: "no text,
   no lettering, no watermark, no logos, no human faces in focus". Difusão lida mal
   com negação no prompt positivo — e pode invocar o que se quer excluir. Se o
   wrapper do modelo no Replicate expuser `negative_prompt`, elas vão para lá.
   (FLUX.1-dev é guidance-distilled e a pipeline base não tem `negative_prompt`;
   alguns wrappers expõem. **Confirmar na página do modelo antes de contar com
   isso** — o plano abaixo torna o dict de input configurável de qualquer forma.)

Um quarto suspeito que ainda não foi medido: **resolução**. O `prep_size` amplia
toda imagem para `4302x2422`. Se o flux devolve ~1344px, viemos ampliando 3.2x
desde o primeiro vídeo, e nenhuma troca de provider resolve isso. O estágio de
render passa a **medir e avisar** (ver "Mudanças de config").

## Forma do pipeline

Três aprovações, **todas antes de gravar**. Depois da gravação, roda sozinho.

    1. script       ideia -> roteiro             script.md          [Opus 5]   APROVA
    2. storyboard   roteiro -> beats visuais     storyboard.json    [Opus 5]   APROVA
    3. assets       beats -> imagens             assets.json        [flux-dev] APROVA
       ------------- você grava seguindo o roteiro -------------
    4. ingest       vídeo -> manifest            manifest.json
    5. transcribe                                transcript.json
    6. align        beats <-> transcript         edl.json           [embeddings locais]
    7. render       sub-planos + overlay         final.mp4

As imagens vêm **antes** de gravar porque estão amarradas a beats do roteiro, não
a segundos. Você revisa com calma; depois de gravar não há mais nada a aprovar.

### O elo que faz isso sobreviver a improviso

O storyboard **nunca escreve segundo absoluto**. Ele escreve um âncora de texto do
roteiro, e o `align` descobre onde aquele beat caiu na fala real, comparando o
âncora com cada segmento do transcript por cosseno — com o `sentence-transformers`
que já roda local para o banco de imagens, de graça e determinístico.

Se você improvisar 8 segundos no meio, nada desalinha: o âncora é semântico, e os
tempos saem sempre do transcript. É o que também preserva "nunca cortar no meio da
frase", porque a fronteira continua sendo fronteira de segmento.

## Schemas entre estágios

### `script.md` (estágio 1)

Markdown, não JSON — é para ler e editar num editor. Cabeçalho YAML com o que os
estágios seguintes consomem, corpo com o roteiro em beats numerados:

```markdown
---
slug: video-sobre-custo
duration_target_seconds: 900
subject: como montar uma esteira de produção de vídeo por menos de R$ 200/mês
argument: a automação já é barata o bastante para sustentar um canal sozinho
viewer_takeaway: dá para montar o estúdio inteiro em software
---

## beat 1 — abertura
Texto do roteiro...

## beat 2 — o problema do custo
Texto do roteiro...
```

O `id` do beat é o número do cabeçalho `## beat N`. Editar o corpo é livre; mexer
na numeração invalida o storyboard (a chave de cache pega isso).

### `storyboard.json` (estágio 2)

```json
{
  "input_hash": "<hash de script.md + regras + style>",
  "beats": [
    {
      "id": 7,
      "script_anchor": "quando eu explico que o custo todo cabe em 200 reais",
      "concept": "a single warm lamp over a wooden desk, a handwritten ledger open beside a coffee cup",
      "concept_tags": ["ledger", "warm lamp", "wooden desk"],
      "sub_shots": 4,
      "seconds_per_shot": 1.5,
      "rationale": "o argumento central precisa de uma imagem de contabilidade caseira"
    }
  ]
}
```

`script_anchor` é **texto do roteiro**, não paráfrase — é o que o alinhamento
compara. `sub_shots` entre 1 e 4 (ver a geometria abaixo). `rationale` existe para
a revisão na UI: sem ele você não sabe por que aquela imagem está ali.

Validação (Pydantic, mesmo padrão do EDL de hoje, 3 tentativas com o erro como
feedback):

- todo `id` existe em `script.md`
- `script_anchor` aparece literalmente no texto daquele beat
- `sub_shots` entre 1 e `render.max_sub_shots`
- soma de `sub_shots * seconds_per_shot` dentro da faixa de cobertura do config
- `concept` em inglês, concreto, sem texto na imagem
- `concept_tags` entre `concept_tags_min` e `concept_tags_max`

### `edl.json` (estágio 6, alinhamento)

Mesma forma de hoje — `EDLSegment` com índices de segmento do transcript — mais o
beat de origem:

```json
{
  "segments": [
    {
      "kind": "broll",
      "beat_id": 7,
      "segment_from": 42,
      "segment_to": 46,
      "start": 187.4,
      "end": 193.4,
      "similarity": 0.81
    }
  ]
}
```

`similarity` fica gravado para a revisão: beat que alinhou fraco (< 0.65) sai no
relatório como suspeito, porque ou você mudou muito o texto ao gravar, ou o âncora
estava ruim.

### `assets.json` (estágio 3, com sub-planos)

```json
{
  "items": [
    {
      "beat_id": 7,
      "image_path": "assets/images/a1b2c3.png",
      "origin": "generated",
      "provider": "flux-dev",
      "source_size": [1344, 768],
      "sub_shots": [
        {"index": 0, "region": "full",         "direction": "zoom_in"},
        {"index": 1, "region": "top_left",     "direction": "pan_right"},
        {"index": 2, "region": "bottom_right", "direction": "zoom_out"},
        {"index": 3, "region": "top_right",    "direction": "pan_left"}
      ]
    }
  ]
}
```

Os sub-planos **não têm tempo aqui** — o tempo vem do `align`, depois. `region` e
`direction` são determinísticos, não escolhidos por modelo. `source_size` é o que
alimenta o aviso de upscale.

## A geometria dos sub-planos

Medido, não estimado:

    prep_size:            4304 x 2420   (saída 1920x1080, canvas 2x, zoom_max 1.12)
    Ken Burns exige:      2151 x 1210   (para nunca ampliar)

    full                 4304 x 2420  -> nativo
    quadrante (metade)   2152 x 1210  -> nativo, exatamente no limite
    um terço             1434 x  806  -> amplia 1.50x

O quadrante bate no limite por construção: `prep_size = largura x canvas_scale x
zoom_max`, então metade disso é `largura x zoom_max` sempre que `canvas_scale = 2`.
A invariante é **canvas_scale 2 <=> quadrante exatamente nativo**, qualquer que
seja o `zoom_max`.

`prep_size` arredonda para cima em múltiplo de **4**, e não de 2, exatamente por
causa disso: metade de um múltiplo de 4 ainda é par, então o `crop` do quadrante
nunca precisa arredondar — e arredondaria para baixo justamente onde a margem é
de 0.6px.

Consequência: **4 sub-planos é o teto real** (os quatro quadrantes). Nove (3x3)
exigiriam `canvas_scale: 3`, ou seja uma imagem de 6453px — nenhum modelo entrega.

Regiões: `full`, `top_left`, `top_right`, `bottom_left`, `bottom_right`.

### O que o quadrante custa (não estava neste plano)

Duas coisas que só apareceram na implementação:

1. **A margem de subpixel do movimento acaba.** No plano cheio o passo de 1px do
   `crop` cai numa tela 2x e vira meio pixel na saída; no quadrante a tela de
   trabalho *já é* a saída. Não aparece porque sub-plano é curto por construção
   (a 2.5s o pan anda 3px por frame); um sub-plano de 10s andaria 0.8px por frame
   e aí o degrau apareceria.
2. **A imagem amplia o dobro na tela.** O fator do `prep` é o mesmo, mas o plano
   cheio é reduzido de volta no fim do Ken Burns e o quadrante não. Uma imagem de
   1344px amplia 1.6x no plano cheio e **3.2x** no quadrante. É por isso que o
   `upscale_warn_factor` mede o quadrante quando o sub-plano está ligado: medir o
   plano cheio diria que 1344px está folgado.

### De onde vem a contagem de planos

O plano previa `sub_shots` gravado em `assets.json`. Na implementação a contagem
sai da **duração da faixa** dividida por `sub_shot_seconds`, e não do número que o
storyboard pediu. O storyboard escolhe de fato o *tempo de tela* do beat
(`sub_shots * seconds_per_shot`); o tamanho exato da faixa só o alinhamento sabe,
e ele cresce em segmento inteiro, então a faixa sobra um pouco. Dividir a faixa
real mantém o **ritmo** que o config pede; usar o número do storyboard manteria a
contagem e esticaria cada plano. O ritmo é o que se vê. Como efeito colateral, o
caminho antigo (EDL vinda do `planner`) ganha sub-planos de graça: uma faixa de
20s vira quatro planos de 5s sem nenhuma imagem nova.

## A UI

FastAPI + HTML estático, local, sem deploy. Nova dependência: `fastapi` e
`uvicorn`. Serve três páginas e monta `work/` e `assets/` como estáticos.

    GET  /                      lista os vídeos em work/, com o estágio de cada um
    GET  /{slug}/script         roteiro renderizado + textarea; salva em script.md
    POST /{slug}/script/approve grava .approved e libera o estágio 2
    GET  /{slug}/storyboard     um cartão por beat: âncora, concept, rationale,
                                sub_shots, e o trecho do roteiro ao lado
    POST /{slug}/storyboard/approve
    GET  /{slug}/assets         contact sheet: miniatura, concept, beat, provider,
                                tamanho da fonte (com aviso se ampliar demais)
    POST /{slug}/assets/{beat}/regenerate   regera UMA imagem
    POST /{slug}/assets/approve

Por que a aprovação é um arquivo (`work/<slug>/script.approved`) e não estado em
memória: o pipeline já é resumível por artefato em disco, e a UI não pode ser o
dono do estado — você tem que poder fechar o navegador, ou rodar tudo pela CLI sem
UI nenhuma.

`pipeline ui` sobe o servidor. A CLI continua funcionando inteira sem ele.

### O que foi construído, e o que ficou de fora

As cinco primeiras rotas existem. As três de `assets` **não**, e não por
economia de esforço: elas precisam de um estágio 3 que ainda não existe. O
`assets` de hoje é indexado por segmento de EDL — que só nasce depois da
gravação — e não por `beat_id`. Uma tela de contact sheet em cima disso
mostraria as imagens do caminho antigo, não as do roteiro que você acabou de
aprovar.

Duas decisões que só apareceram construindo:

- **Nenhum GET gasta dinheiro.** A tela do storyboard não chama o agente: se
  `storyboard.json` não existe, ela diz qual comando rodar. Navegador
  recarrega por conta própria, e prefetch de link também — uma página que
  dispara uma chamada de Opus ao ser aberta é uma armadilha. Pela mesma razão,
  `pipeline approve storyboard` passou a ler o artefato em vez de chamar
  `stages.storyboard.run`.
- **Salvar valida antes de gravar, e grava verbatim.** O texto do textarea é
  parseado primeiro; se não passar, a página volta com o erro e com o texto
  intacto no campo. Passando, é gravado byte a byte como veio — e não o
  `render()` do que foi parseado, que normalizaria o frontmatter e descartaria
  uma nota antes do primeiro beat.

O estado dos portões é lido em um lugar só (`progress.py`), usado pela CLI e
pela UI. Duplicar seria a divergência que não falha: a UI dizendo "aprovado" e
o `status` dizendo "editado!".

### O que ainda falta para o caminho rodar de ponta a ponta

Dois estágios, não a UI:

1. **estágio 3, `assets` por beat.** Hoje o resolver junta imagem e vídeo por
   índice de segmento da EDL. Antes de gravar não existe EDL, então a chave
   precisa ser `beat_id`.
2. **estágio 6, `align` como estágio.** O módulo `align.py` existe e está
   testado, mas nada escreve `edl.json` a partir de storyboard + transcript.
   `stages.SCRIPT_FIRST` reflete isso: ele não lista `assets` nem `align`.

Os dois juntos levantam uma pergunta de desenho que não está resolvida neste
plano: quem traduz `beat_id` para índice de segmento, e onde isso fica
gravado. A EDL nasce no `align`, que é o único ponto que conhece as duas
coisas — o que sugere um `assets.aligned.json` (como o
`transcript.trimmed.json` já faz) para não reescrever um artefato que já foi
aprovado. **Isso precisa ser decidido antes de ser escrito.**

## Mudanças de config

```yaml
image_provider:
  active: replicate
  replicate:
    model: black-forest-labs/flux-dev     # era flux-schnell
    cost_usd_per_image: 0.025             # era 0.003
    # O dict de input do provider passa a ser configurável: hoje ele manda
    # `num_outputs` hardcoded, que nem todo modelo aceita. Isso é o que torna a
    # promessa de "provider plugável" verdadeira.
    input:
      aspect_ratio: "16:9"
      output_format: png
      num_outputs: 1

budget:
  # 90 imagens a $0.025 dão $2.25/vídeo. Com o teto em 1.50 o estágio 3 PARA e
  # mostra a estimativa — que é o comportamento certo, mas precisa subir.
  max_usd_per_video: 3.00

render:
  max_sub_shots: 4          # teto de geometria, não de gosto
  sub_shot_seconds: 1.5
  # Novo: avisa quando a imagem gerada é pequena demais para o prep_size e
  # precisa ser ampliada. É o que responde se "delírio" é aderência ou resolução.
  upscale_warn_factor: 1.5
```

## Custo por vídeo de 15 min (60% de cobertura = 540s de b-roll)

| sub-plano | planos | imagens | flux-dev | por mês (8 vídeos) | cabe em $1.50? |
|---|---|---|---|---|---|
| 1.5s | 360 | 90 | $2.25 | R$ 97 | não, subir para $3 |
| 2.0s | 270 | 68 | $1.70 | R$ 73 | não |
| 2.5s | 216 | 54 | $1.35 | R$ 58 | sim |
| 3.0s | 180 | 45 | $1.13 | R$ 49 | sim |

Mais ~$0.20/vídeo de Opus 5 (roteiro + storyboard), ou ~R$ 9/mês.

Total: **R$ 58 a R$ 106/mês**, contra o teto de R$ 200. Cabe em qualquer linha.

## O que fica, o que sai

**Fica, sem mexer:** ingest, transcribe, o banco de assets com isolamento por
estilo e modelo de embedding, o `filtergraph` inteiro (overlay, Ken Burns,
chunking, fade proporcional), o corte de pausa e hesitação, os 244 testes.

**Sai:** o `planner.py` de duas fases. O briefing visual e a EDL a partir do
transcript são substituídos por roteiro + storyboard + alinhamento. O código de
validação (`edl.py`) fica, porque a EDL continua existindo — só passa a ser
derivada do alinhamento em vez de gerada por modelo.

**Muda:** `resolver.py` ganha sub-planos; `stages/render.py` expande sub-plano em
overlay; `providers.py` ganha o dict de input configurável.

## Testes

- alinhamento: beat cujo texto foi improvisado ainda casa; beat que não existe na
  fala sai como não alinhado em vez de casar errado
- geometria: quadrante é nativo; um terço é rejeitado; `max_sub_shots` respeitado
- sub-plano no render: smoke com ffmpeg medindo que os 4 recortes mostram regiões
  **diferentes** da mesma imagem (comparar brilho/hash de cada plano)
- storyboard: âncora que não existe no roteiro é rejeitado e volta como feedback
- aprovação: estágio 2 não roda sem `script.approved`
- UI: cada rota responde; regenerar uma imagem não toca nas outras

## Riscos e o que ainda não sei

1. **`negative_prompt` no flux-dev do Replicate** — não consegui confirmar (egresso
   bloqueado). Se não existir, as negações continuam no prompt positivo e o conserto
   nº 3 cai.
2. **Resolução da saída do flux-dev.** Se vier ~1344px, o upscale de 3.2x continua e
   pode ser o verdadeiro culpado do "delírio" percebido. O aviso novo mede isso no
   primeiro run.
3. **Alinhamento de beat que você cortou ao gravar.** Se você pular um beat inteiro,
   a imagem dele fica órfã. O plano: sai no relatório e o segmento vira a-roll — não
   inventa lugar para ela.
4. **`script_anchor` literal** é uma exigência forte no validador. Se o modelo
   parafrasear, gasta tentativa. Alternativa se incomodar: aceitar âncora por
   similaridade em vez de literal, ao custo de alinhamento mais frouxo.
