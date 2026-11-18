# Modo servidor — um VisionForge para várias pessoas (experimental)

> **Leia antes de instalar.** O modo servidor está em construção e **ainda não
> serve para treinar**: nesta versão ele só identifica quem entrou e mostra o
> histórico e os tokens de cada pessoa. As telas de dataset, o treino, a fila e
> o Colab respondem "não disponível" até serem revistas uma a uma. E ele é para
> um **grupo que já confia entre si**: quando o treino for liberado, um membro
> poderá executar código no servidor (um checkpoint de detecção é um arquivo
> pickle). Não ofereça este servidor a desconhecidos.

Esta página sai do código atual (`gui/server_mode.py`) e das decisões ADR-212
e ADR-213. O que não foi testado está marcado como *não verificado*.

## O que muda em relação ao uso local

No seu computador, o VisionForge confia em quem abre a página: o perfil é uma
escolha do navegador. Num servidor isso não basta, porque qualquer pessoa
poderia escolher o perfil de outra. No modo servidor:

- **O login fica num proxy.** O VisionForge não tem tela de senha. Um proxy
  reverso que faz login (oauth2-proxy, Authelia) fica na frente, com a conta do
  GitHub, do Google ou da instituição, e diz ao VisionForge quem entrou.
- **Só o proxy é atendido.** Um pedido que chega direto à porta do VisionForge,
  sem passar pelo proxy, é recusado (403).
- **Cada conta tem a sua pasta.** O nome que o proxy informa vira o perfil da
  pessoa, criado no primeiro acesso em `outputs/profiles/`. O que o navegador
  mandar como perfil é ignorado.
- **O que não está pronto responde 404**, com a mensagem "Esta função não está
  disponível quando o VisionForge roda como servidor".

## Como ligar

```powershell
visionforge gui --server --trusted-proxy 127.0.0.1 --allowed-host visionforge.meulab.org
```

| Opção | Para que serve |
| --- | --- |
| `--server` | Liga o modo servidor. Sem ela nada muda. |
| `--trusted-proxy <endereço>` | IP (ou bloco CIDR, como `172.18.0.0/16`) do proxy de login. Obrigatória: sem ela o servidor não sobe. Pode repetir. |
| `--user-header <nome>` | Cabeçalho em que o proxy informa a conta. O padrão é `X-Forwarded-User` (oauth2-proxy); no Authelia é `Remote-User`. |
| `--allowed-host <nome>` | O nome público pelo qual as pessoas chegam (o do proxy). |

Deixe o VisionForge escutando só em `127.0.0.1` (o padrão) quando o proxy roda
na mesma máquina. Se o proxy roda em outra máquina ou em outro contêiner, use
`--host` com o endereço da rede interna e feche a porta para o resto no
firewall.

### O que o proxy precisa fazer

1. Exigir login em **todas** as rotas, menos `GET /api/health` (que serve para
   o proxy saber se o VisionForge está de pé).
2. Enviar a conta autenticada no cabeçalho combinado e **apagar** esse
   cabeçalho de qualquer pedido que venha do navegador. O oauth2-proxy e o
   Authelia fazem isso por padrão; num proxy montado à mão, confira.
3. Servir por HTTPS.

*Não verificado:* nenhuma instalação real com oauth2-proxy ou Authelia foi
feita ainda; o comportamento acima foi testado com pedidos simulados.

## O que responde hoje

| Função | No modo servidor |
| --- | --- |
| Abrir a página, esquemas dos formulários, informações do sistema | sim |
| Histórico: listar, abrir e apagar os **seus** runs | sim |
| Tokens de publicação (Hugging Face, GitHub, Zenodo, W&B, Comet) do **seu** perfil | sim |
| `GET /api/session` (quem entrou) | sim |
| Telas de dataset e Laboratório de datasets | ainda não (precisam ficar presas à pasta de cada pessoa) |
| Treinar, estratégias, avaliar, exportar | ainda não (cada rota será revista) |
| Fila e progresso do treino | ainda não (hoje mostram os jobs de todos) |
| Seletor de pasta, "abrir pasta", atualizar o VisionForge, Colab, tarefas próprias, credenciais de download | desligados no modo servidor |

A interface ainda não sabe do modo servidor: ela mostra botões que respondem
404. Isso muda quando a tela passar a ler `GET /api/session`.

## Perguntas comuns

**Os runs antigos do perfil "default" aparecem?** Não. No modo servidor
ninguém usa o perfil padrão; cada conta tem a própria pasta. Os runs antigos
continuam no disco, em `outputs/models/`.

**Duas contas parecidas podem cair na mesma pasta?** Não. O nome da pasta
termina com um resumo (hash) do nome completo da conta, então `a.b@lab.org` e
`a-b@lab.org` ficam separadas. Maiúsculas não contam: `Ana@lab.org` e
`ana@lab.org` são a mesma conta.

**Os tokens ficam cifrados?** Ainda não: são arquivos de texto na pasta de
configuração do usuário do sistema que roda o servidor (veja [Tokens](tokens.md)).
Quem administra a máquina consegue lê-los.
