# Aplicação da avaliação com participantes externos

Este roteiro operacionaliza `PROTOCOL.md` e `FORM.md`; não registra participantes
nem resultados. O protocolo existente continua sendo a fonte dos instrumentos
e das limitações de interpretação.

## Preparação pelo pesquisador

1. Confirmar com orientador/instituição as condições de participação e uso dos
   dados antes do recrutamento, conforme a pendência já registrada no protocolo.
2. Fixar versão do VisionForge, release/commit, README e formulário usados.
3. Fazer uma sessão piloto para testar instruções e cronômetros; marcar o piloto
   separadamente e registrar qualquer mudança antes da primeira coleta válida.
4. Entregar somente o material previsto no protocolo. Registrar ajuda prestada
   durante as tarefas, em vez de corrigir silenciosamente a dificuldade.
5. Usar identificadores aleatórios de sessão. Não coletar nomes, e-mails,
   caminhos pessoais completos, tokens, imagens privadas ou gravações de tela
   por padrão. Revisar texto livre antes de compartilhar resultados.

## Registro mínimo por sessão

O arquivo `session_template.csv` contém apenas o cabeçalho. Repetir uma linha
por tarefa T1–T5 e manter o mesmo `session_id` aleatório na sessão.

- `outcome`: `completed`, `assisted`, `failed` ou `withdrawn`.
- `elapsed_seconds`: duração observada, inclusive em falhas; não preencher um
  tempo estimado para tarefa que não foi medida.
- `assistance_count`: quantidade de intervenções do facilitador.
- `issue_category`: instalação, compreensão, dados, configuração, execução,
  resultados ou outra categoria explícita.
- `notes_redacted`: descrição revisada sem dados identificáveis.

SUS e quiz permanecem no formulário: usar `sus.py` para pontuação e respeitar
o pareamento de pré/pós. Não transformar SUS em porcentagem de sucesso.

## Depois da sessão

Registrar versão e contexto técnico geral, revisar anonimização, preservar
falhas e desistências e executar a análise descritiva prevista. O pesquisador
decide sobre alterações do produto a partir dos problemas observados; mudanças
durante a coleta devem ser registradas como desvios, sem misturar versões como
se fossem a mesma condição.

Não afirmar efeito causal das animações com o desenho atual sem grupo de
controle. Não preencher resultados fictícios para ilustrar o artigo.
