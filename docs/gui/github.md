# Enviar o relatório de uma execução ao GitHub

No **Histórico**, **Enviar o relatório ao GitHub** grava a execução num
repositório **privado que já existe**, em um único commit (ADR-175). Informe o
repositório no formato `usuário/repositório`. A prévia é local: mostra a
pasta, cada arquivo com o tamanho e o README, e o commit só sai se bater com
essa prévia.

## O que é enviado

Uma pasta `runs/<experimento>/<execução>/` com:

- um README gerado do `run.json`: resumo, métricas finais, uma tabela por
  época (no máximo 200 linhas) e os gráficos embutidos;
- o `run.json` público, sem caminhos da máquina e com `artifacts.model` nulo;
- os gráficos em `figures/` (PNG e JPEG, até 5 MB cada, no máximo 20).

Os arquivos entram num commit só, na branch padrão, sem `force`. Num
repositório vazio, o README vai primeiro pela API de conteúdo, a única forma
de fazer o primeiro commit.

## O que não é enviado

- **Pesos, nunca.** Eles têm outros destinos: Hugging Face Hub e Zenodo.
- O dataset.
- Os caminhos locais do `run.json`.
- Épocas além da 200ª na tabela do README (o `run.json` guarda todas).
- Um repositório novo: o VisionForge **não cria** repositório.

## Token e escopo mínimo

Token fine-grained, limitado ao repositório de experimentos, com a permissão
**Contents: leitura e escrita**. Ele fica salvo por perfil e vai só para
`https://api.github.com`. O repositório é lido antes de qualquer escrita e,
se não for privado, o envio é recusado. Só github.com; GitHub Enterprise não é
suportado. Detalhes: [Tokens e chaves](https://github.com/marcus-vreis/VisionForge/blob/development/docs/gui/tokens.md).

## Conectar com GitHub, sem colar token

Quando quem opera o servidor registrou um *OAuth App* do GitHub e passou as
credenciais ao VisionForge (veja abaixo), o painel mostra **Conectar com
GitHub** ao lado do campo do token (ADR-250). O botão abre o GitHub, você
autoriza o app e volta para a página; o token fica salvo neste perfil, no
mesmo lugar de um token colado, e o envio funciona igual. Ao voltar, um aviso
acima da barra de baixo diz se deu certo ("GitHub conectado: publicar um run
usa essa conta") ou por que não (recusado no GitHub, ou o GitHub não devolveu
token); recarregar a página não repete o aviso. **Remover token**
desconecta. Colar um token continua funcionando, com ou sem o botão.

- **Escopo `repo`.** O envio só escreve em repositório privado, e o escopo
  `public_repo` só alcança repositórios públicos; por isso o app pede `repo`,
  que dá escrita em **todos** os seus repositórios, privados inclusive. Para
  limitar a um repositório só, cole um token fine-grained em vez de conectar.
- **Segurança.** O pedido leva um `state` aleatório que vale 10 minutos e uma
  vez só, ligado ao seu perfil, e usa PKCE (`S256`). O código e o token nunca
  aparecem na URL de volta nem no log.

### Como o operador liga

1. No GitHub, **Settings → Developer settings → OAuth Apps → New OAuth App**.
   Em *Authorization callback URL*, o endereço público do VisionForge seguido
   de `/api/github/oauth/callback` (por exemplo,
   `https://visionforge.meulab.org/api/github/oauth/callback`; no uso local,
   `http://127.0.0.1:8000/api/github/oauth/callback`).
2. Gere um *client secret* e, antes de subir o servidor, defina no ambiente
   (nunca na linha de comando, que outros usuários da máquina conseguem ver):

   ```powershell
   $env:VISIONFORGE_GITHUB_CLIENT_ID = "<client id>"
   $env:VISIONFORGE_GITHUB_CLIENT_SECRET = "<client secret>"
   # Opcional: o endereço público, para mandar o redirect_uri ao GitHub.
   $env:VISIONFORGE_PUBLIC_URL = "https://visionforge.meulab.org"
   ```

3. Sem `VISIONFORGE_PUBLIC_URL`, o VisionForge não manda `redirect_uri` e o
   GitHub usa o callback cadastrado no app. Com ela, o endereço tem de ser
   `http(s)://` sem `?` nem `#`, e precisa bater com o cadastrado; um valor
   inválido desliga o botão e deixa um aviso no log.

Sem as duas variáveis do app, o botão não aparece e as rotas
`/api/github/oauth/*` respondem 404. O mesmo vale no uso local e no
[modo servidor](https://github.com/marcus-vreis/VisionForge/blob/development/docs/gui/servidor.md).

## Se falhar, e repetir

A resposta do GitHub nunca aparece. Os erros 401, 403 e 404 têm mensagem
própria (o 404 também significa que o token não enxerga um repositório
privado); o resto é uma falha genérica. Enviar de novo troca os arquivos da
pasta da execução em **um commit novo**. Arquivos que saíram da execução
continuam no repositório até você apagá-los.

## O que não foi verificado

As chamadas à API (blobs, árvore, commit, atualização da referência) estão
cobertas por testes com simulação; **não foram conferidas com uma conta real**
do GitHub. O fluxo **Conectar com GitHub** também foi testado só com o GitHub
simulado na camada HTTP: nenhum OAuth App real foi autorizado.

Decisão e motivos: ADR-175 e ADR-250 em [`DECISIONS.md`](https://github.com/marcus-vreis/VisionForge/blob/development/docs/dev/DECISIONS.md).
