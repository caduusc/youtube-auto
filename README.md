# youtube-auto

Pega um arquivo único de você falando por 10 a 15 minutos e devolve um vídeo
montado onde partes da sua imagem são cobertas por b-roll ilustrativo, com
legenda queimada e **o áudio original intacto do começo ao fim**.

Seis estágios, cada um executável isoladamente, cada um lendo e escrevendo
artefatos em `work/<slug>/`. O pipeline é resumível: se você matar o processo
no estágio 5, rodar de novo retoma do 5 sem refazer transcrição nem regerar
imagem.

```
ingest ─> transcribe ─> trim ─> plan ─> assets ─> render ─> report
   │           │          │       │        │         │         │
manifest  transcript    trim     edl    assets   final.mp4  report
  .json      .json      .json   .json    .json               .json
                          └─ transcript.trimmed.json
```

## Setup

```bash
git clone <este-repo> && cd youtube-auto
uv sync                        # Python 3.11+
cp config.example.yaml config.yaml
```

Você também precisa de **ffmpeg e ffprobe** no PATH (`apt install ffmpeg`,
`brew install ffmpeg`).

O primeiro `run` baixa dois modelos locais, uns 250MB no total: o
`faster-whisper` da transcrição e o de embedding do banco de assets. Fica
tudo em cache no seu `~`, então é uma vez só.

O modelo de embedding é **inglês**, apesar de o vídeo ser em português. O
banco só embeda o campo `concept`, que o estágio 3 sempre escreve em inglês;
o transcript nunca passa por ali. Um modelo multilíngue custaria 471MB de
download para nenhum ganho.

Se o download do modelo de embedding falhar (`CAS Client Error`,
`error decoding response body`, `DECRYPTION_FAILED_OR_BAD_RECORD_MAC`), é
transferência do HuggingFace, não o pipeline: rodar de novo retoma de onde
parou. Dois contornos, em ordem:

- `set HF_HUB_DISABLE_XET=1` troca para o download clássico, mais tolerante;
- se persistir, baixe fora do pipeline com `hf download <modelo>`, que
  retenta sozinho, e depois rode normalmente — o cache é o mesmo.

**Erro de TLS** (`DECRYPTION_FAILED_OR_BAD_RECORD_MAC`) é caso diferente:
não é falta de rede, é algo quebrando os registros da conexão. Reexecutar não
resolve — arquivos pequenos passam e a transferência sustentada quebra. Em
ordem de eficácia:

1. `uv pip install hf_transfer` e `set HF_HUB_ENABLE_HF_TRANSFER=1` — o
   downloader em Rust usa outra pilha TLS;
2. desligar a inspeção de HTTPS do antivírus ou a VPN durante o download, que
   é a causa mais comum;
3. `hf download <modelo>` repetido até completar, já que cada tentativa retoma
   de onde parou;
4. baixar os arquivos do modelo pelo navegador e apontar
   `bank.embedding_model` para a pasta local — o campo aceita caminho,
   resolvido contra a raiz do projeto.

Para o item 4 com `all-MiniLM-L6-v2`, são 10 arquivos (~91MB): `config.json`,
`config_sentence_transformers.json`, `modules.json`, `sentence_bert_config.json`,
`special_tokens_map.json`, `model.safetensors`, `tokenizer.json`,
`tokenizer_config.json`, `vocab.txt` e `1_Pooling/config.json`. **Não** baixe
`pytorch_model.bin` nem `rust_model.ot` — são os mesmos pesos em outros
formatos. Verifique a pasta antes de rodar o pipeline:

```bash
uv run python -c "from sentence_transformers import SentenceTransformer; \
  print(SentenceTransformer('models/all-MiniLM-L6-v2').encode('teste')[:4])"
```

O estágio 4 distingue os dois modos de falha e imprime essa lista quando o
erro é de TLS.

O estágio 4 carrega esse modelo **antes** de baixar ou gerar qualquer imagem,
justamente para que essa falha não aconteça depois de você já ter pago.

**Trocar `bank.embedding_model` invalida o banco.** Os embeddings de modelos
diferentes não são comparáveis, e como vários têm as mesmas 384 dimensões a
incompatibilidade não apareceria como erro — apareceria como similaridade sem
sentido. O banco guarda qual modelo gerou cada linha e só compara dentro do
mesmo; `bank stats` mostra as famílias quando há mais de uma.

### Variáveis de ambiente

Nenhuma chave vai para o `config.yaml` — ele guarda só o *nome* da variável
de onde cada chave é lida.

| Variável | Para quê | Sem ela |
|---|---|---|
| `ANTHROPIC_API_KEY` | estágio 3, planejamento editorial | o pipeline não roda |
| `REPLICATE_API_TOKEN` | estágio 4, geração de imagem | `--dry-run` funciona; o run real falha, a menos que você ponha `image_provider.active: none` |
| `PEXELS_API_KEY` | estágio 4, stock gratuito | tudo que seria stock vai para geração, e a conta sobe |

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export REPLICATE_API_TOKEN=r8_...
export PEXELS_API_KEY=...          # gratuito em pexels.com/api
```

## O primeiro vídeo

### 1. Calibre antes de gastar

```bash
uv run pipeline run aula.mp4 --dry-run
```

Isso roda ingest, transcribe e plan de verdade, e vai até o estágio 4 **sem
gastar um centavo**: mostra a EDL inteira, quantos segmentos viriam do banco,
quantos do stock, quantos seriam gerados, o prompt exato de cada geração e o
custo estimado em USD e BRL.

É o comando que você vai usar antes de cada vídeo. Duas coisas para olhar:

- **os `concept`**. Eles descrevem o quadro que a imagem vai mostrar. Se
  estiverem abstratos ("the idea of growth") em vez de concretos ("a single
  green shoot pushing through cracked asphalt"), o resultado vai ser genérico.
  O que conserta isso é o system prompt em `src/pipeline/planner.py`.
- **a divisão de custo**. Se está gerando quase tudo, amplie
  `stock.generic_tags` no config. Cada tag que casa é uma imagem que sai de
  graça.

O dry-run é barato de repetir: transcribe e plan ficam em cache, então da
segunda vez em diante ele responde na hora.

### 2. Rode de verdade

```bash
uv run pipeline run aula.mp4
```

A saída fica em `work/<slug>/final.mp4` e o relatório em `report.json`.

### 3. Acompanhe o banco

```bash
uv run pipeline bank stats
```

A taxa de reuso é o que faz a conta fechar. No primeiro vídeo ela é 0% e
todo b-roll é gerado. Do terceiro ou quarto em diante, se os seus vídeos
falam de assuntos próximos, boa parte dos segmentos passa a ser resolvida de
graça pelo banco.

```bash
uv run pipeline bank prune --unused-days 90
```

## Conexão instável

`network.download_attempts` (default 6) existe porque numa conexão que
corrompe TLS em transferência sustentada, as 3 tentativas do spec dão ~50% de
chance de perder um run de 20 imagens. Seis derrubam para ~3%.

A assimetria com `api_attempts` (3) é deliberada: baixar por URL é idempotente
e de graça, mas criar uma predicação no provider **cobra** — se ela roda no
servidor e a resposta se perde, repetir gera uma segunda imagem e cobra duas
vezes.

Downloads escrevem num `.part` e só movem para o destino ao completar, então
uma queda no meio não deixa arquivo truncado que o resto do pipeline trataria
como imagem válida.

## Rodando sem conta de geração

Se você não tem crédito no Replicate, `image_provider.active: none` no config
faz o pipeline usar só o banco e o stock gratuito. O que nenhum dos dois
resolver recebe o fallback de cor sólida, e o vídeo sai — custo de imagem
zero.

O limite é real: stock só cobre cena genérica. Um `concept` específico
("a chessboard mid-game beside a window") não existe em banco de fotos, e
aquele segmento vira um retângulo de cor. Serve para validar o pipeline
inteiro e para vídeo cujo b-roll é todo genérico; não substitui geração.

Ampliar `stock.generic_tags` é o que aumenta a cobertura nesse modo: cada tag
que casa é um segmento que sai do Pexels em vez de virar cor sólida.

## Orçamento

R$ 200/mês para 8 vídeos dá R$ 25 por vídeo, uns USD 4,60. O default de
`budget.max_usd_per_video` é **USD 1,50**, de propósito bem abaixo disso.

O teto age em dois momentos:

- **antes de começar**, o estágio 4 estima o pior caso (tudo gerado). Se
  estourar, ele para e mostra o relatório de estimativa em vez de gastar;
- **durante**, um acumulador soma o custo real devolvido pelo provider. O que
  passar do teto fica marcado para fallback de cor sólida em vez de gerar.

Com FLUX schnell a USD 0,003 por imagem e uns 20 segmentos de b-roll, o pior
caso de um vídeo é USD 0,06. A folga é grande — se você quiser mais qualidade
por imagem, troque `image_provider.replicate.model` para
`black-forest-labs/flux-dev` e ajuste `cost_usd_per_image` para `0.025`. O
pior caso passa a USD 0,50, ainda bem dentro.

O planejamento editorial usa Claude Opus 5 e custa cerca de USD 0,30 por
vídeo — mais que as imagens. É deliberado: a EDL é o que determina se o
vídeo fica bom.

## Rodando um estágio isolado

```bash
uv run pipeline transcribe aula-a1b2c3d4     # o slug sai do nome do arquivo + hash
uv run pipeline plan       aula-a1b2c3d4
uv run pipeline render     aula-a1b2c3d4
```

Ou retome o run completo de um estágio:

```bash
uv run pipeline run aula.mp4 --from render
```

Cada artefato carrega o hash daquilo de que foi derivado, então um estágio
sabe sozinho se o trabalho dele já está feito.

O hash inclui o config que importa para aquele estágio: mexer em `editorial`
invalida a EDL, e mexer em `style_suffix`, `similarity_threshold`,
`generic_tags` ou no modelo do provider invalida os assets. Ou seja, calibrar
funciona — você edita o `config.yaml`, roda de novo, e o estágio afetado
refaz o trabalho sozinho sem refazer a transcrição.

Para forçar a refazer mesmo sem mudar o config, apague o artefato:

```bash
rm work/aula-a1b2c3d4/edl.json && uv run pipeline plan aula-a1b2c3d4
```

## Como as decisões foram tomadas

### O áudio nunca é filtrado

A trilha de vídeo é a única coisa que passa por filtro; o áudio sai do
arquivo original por stream copy. Isso exige que a duração da saída seja
exatamente a da entrada.

Por isso a transição **não** usa `xfade`: ele consome o overlap, encurtando a
saída em 400ms por transição. Com 20 transições o vídeo sairia 8 segundos
mais curto que o áudio.

O que o pipeline faz: o vídeo original é a camada base, rodando inteiro com o
timing intacto, e cada imagem de b-roll é sobreposta por cima com fade de
alpha na entrada e na saída. O fade das duas pontas *é* o crossfade nas duas
direções, e a duração não muda.

Uma consequência: os 400ms de fade vivem dentro do próprio segmento de
b-roll, então um b-roll de 8s fica 7,2s em opacidade cheia.

### Ken Burns sem `zoompan`

O `zoompan` calcula o recorte em pixels inteiros. Num pan lento a janela
deveria andar meio pixel por frame, então ela anda 1px a cada dois frames — e
esse degrau é o tremor.

Aqui o movimento sai de dois filtros que o ffmpeg reavalia por frame: `scale`
com `eval=frame` para o zoom, e as expressões x/y do `crop` para a
translação. Ambos operam numa tela de 2x a resolução de saída, e o downscale
para 1080p vem por último — então o passo de 1px vira meio pixel na saída.

Nos pans o zoom é constante, então o `scale` recebe tamanho fixo e o link nem
reconfigura; só o `x` do crop varia.

**O que foi medido, e o que não foi.** Este caminho foi executado no ffmpeg
6.1: num pan de 1.12 sobre 2s, 58 de 58 frames mudaram, com diferença regular
entre frames (CV 0.05). Não há aqui uma medição do `zoompan` tremendo — o
caso que provoca o artefato é mais lento que o que eu medi, e no pan que
testei o `zoompan` corrigido também saiu liso. O que sustenta a escolha é o
mecanismo, não um número comparativo. Se você notar tremor no resultado real,
vale medir os dois antes de concluir de qual lado está o problema.

Se a sua build de ffmpeg recusar a reconfiguração por frame,
`render.ken_burns.engine: zoompan` no config troca o motor. Cuidado: o `d` do
zoompan conta frames de saída por frame de **entrada**, e a entrada é uma
imagem em `-loop 1`, então o chain precisa do `select='eq(n\,0)'` na frente
para não multiplicar a duração. Isso já está no código.

### A EDL vem em índices, não em tempos

"Nunca cortar para b-roll no meio de uma frase" poderia ser uma regra
validada por tolerância sobre tempos. Em vez disso, o modelo devolve faixas
de **índice de segmento do transcript**, e o pipeline deriva os tempos. A
fronteira passa a ser, por construção, fronteira de segmento: cortar no meio
de uma frase deixa de ser representável.

### O corte seco e a regra que mais importa é negativa

O estágio `trim` remove pausa longa e hesitação. Em português, vários sons de
hesitação são palavras de conteúdo: **"é" é a 3ª pessoa de "ser"** e "um" é
artigo. Cortar por token destruiria frases.

Todo candidato passa por três testes ao mesmo tempo: estar na lista de
`fillers`, durar acima do mínimo, e ter silêncio dos **dois** lados. Exigir os
dois lados em vez de um é o que protege a pausa retórica — em *"o problema
é... que ninguém olha"*, a pausa depois do verbo bastaria para marcá-lo como
hesitação se um lado fosse suficiente.

Os cortes são listados com o texto ao redor no `--dry-run` e em `trim.json`.
Sem isso o corte é caixa preta: você vê 12% de redução e não tem como saber
se uma pausa que dava peso a uma frase foi embora.

**O alinhamento a frame não é cosmético.** `atrim` corta áudio por amostra e
`trim` corta vídeo por frame: em tempo arbitrário, cada corte deixa até um
frame de diferença entre as trilhas. Medido com 20 cortes não alinhados: 24ms
de dessincronia acumulada. Com os trechos alinhados ao grid: zero.

Com corte, o áudio deixa de ser byte a byte igual — ele é cortado nos mesmos
instantes do vídeo, o que preserva a sincronia labial, e encodado uma vez. O
invariante passa a ser "cortado nos pontos escolhidos e nunca processado de
outra forma": sem normalização, sem compressão, sem filtro. Com
`trim.enabled: false`, volta ao stream copy.

### O planejamento tem duas fases

A fase 1 lê a transcrição inteira e devolve um briefing visual: assunto,
argumento, vocabulário visual e clichês a evitar. A fase 2 monta a EDL
escolhendo dentro daquele vocabulário.

Existe porque as duas tarefas competiam na mesma resposta. Pedir na mesma
chamada "segmente a timeline" e "invente as imagens" fazia a segunda sofrer:
saíam cenas plausíveis para um vídeo de criador genericamente, não para
*este* vídeo.

O briefing define o **vocabulário**; cada `concept` continua ancorado ao
próprio trecho. Uma imagem que ilustra o tema geral mas não o que está sendo
dito naquele momento é pior que uma imagem genérica — o espectador sente o
descolamento entre o que ouve e o que vê.

Custa uma chamada a mais por vídeo, uns US$ 0,05. `anthropic.two_phase:
false` volta ao passe único.

### As regras editoriais interagem de um jeito que não é óbvio

Com faixas de b-roll de `L` segundos e proporção `r`, a taxa de trocas é
`120 * r / L` — **não depende da duração do vídeo**. Com os defaults
(`L ≤ 25s`, 3 trocas/min), a proporção para de subir em `25*3/120 = 62,5%`:
o teto de `broll_ratio_max: 0.70` nunca é atingido.

Isso importa na prática: um modelo que mire no meio da faixa nominal 50-70%
viola o ritmo em toda tentativa. Por isso o estágio 3 calcula a janela real
para o vídeo em questão e a manda no prompt, em segundos absolutos, com
precedência sobre as faixas relativas.

Vídeo curto aperta muito mais. Em 2 minutos só existe **uma** configuração
válida: 3 faixas de b-roll de ~23s cada. Abaixo de ~70 segundos não existe
nenhuma — a intro obrigatória de 20s não cabe junto com 50% de b-roll. O
estágio 3 detecta isso antes de chamar a API e diz qual knob soltar, em vez
de gastar três tentativas de Opus descobrindo.

Para testar com vídeo curto, baixe `intro_aroll_seconds` para uns 8s e
`broll_ratio_min` para uns 0.35.

### "3 trocas por minuto" foi separado em dois números

Como janela deslizante estrita, a regra é inviável junto com as outras duas:
b-roll de no máximo 25s cobrindo pelo menos 50% do vídeo força um ciclo de no
máximo 50s, e qualquer padrão com ciclo abaixo de 60s tem alguma janela de
60s com 4 fronteiras dentro.

Então a taxa de 3/min vale na média do vídeo inteiro, e a janela deslizante
ganha a folga de 1 que o efeito de borda produz. Isso ainda barra rajada
local: b-roll de 8s picado com a-roll curto estoura as duas contas.

`src/pipeline/edl.py` tem a conta; `tests/test_edl_validation.py` fixa o
comportamento.

### O filtergraph é escrito em disco antes de rodar

Em `work/<slug>/filtergraph.txt`. Se o render falhar, é o primeiro lugar para
olhar. Acima de `render.max_filtergraph_chars` (3000) a renderização é
quebrada em chunks por janela de tempo e concatenada com o demuxer concat —
os cortes caem no início de um b-roll, que é sempre a-roll puro, então
nenhuma transição atravessa a fronteira de um chunk.

Num vídeo de 15 minutos com ~20 segmentos, o grafo passa de 6000 caracteres,
então o caminho de chunks é o normal, não a exceção.

## Testes

```bash
uv run pytest
```

Cobrem a validação da EDL, a lógica de reuso do banco, o loop de rejeição do
estágio 3, a análise de viabilidade e a geração do filtergraph. Toda chamada externa é mockada:
**nenhum teste gasta dinheiro** e nenhum precisa de chave de API, de ffmpeg
ou dos modelos locais.

## O campo mais importante do config

`style_suffix`. Ele é concatenado ao `concept` de cada segmento antes de ir
para o provider, e é o único mecanismo que faz 20 imagens geradas em 20
chamadas independentes parecerem sair da mão do mesmo ilustrador.

Calibre com `--dry-run` até gostar, e depois **não mexa mais** — ou os vídeos
antigos e novos vão parecer de canais diferentes.
