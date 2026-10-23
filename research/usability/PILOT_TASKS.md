# Piloto antes da coleta válida

Este roteiro serve para encontrar bloqueios e ajustar instruções. Suas sessões
não entram automaticamente na amostra do protocolo registrado em PROTOCOL.md.
Não substitui as tarefas T1–T5 nem mede efeito causal do guia.

## Preparação

- Fixar commit, wheel e seu SHA-256; conferir a versão e o caminho do pacote
  instalado. Não misturar uma GUI de development com o pacote publicado anterior.
- Usar máquina e dados de teste separados de experimentos em andamento. No piloto
  inicial, selecionar CPU, zero workers e pesos aleatórios, sem downloads.
- Preparar um dataset pequeno com classes em train/val/test, uma cópia com uma
  duplicata exata entre splits e um CSV de grupos conhecido pelo facilitador.
- Confirmar as condições institucionais de participação previstas no protocolo;
  usar identificadores anônimos e registrar ajuda, falhas e desistências.

## Tarefas adicionais de exploração

1. Instalar o wheel fornecido e abrir a GUI seguindo somente as instruções.
2. Selecionar a cópia do dataset com duplicata, executar a auditoria e explicar
   o achado. Perguntar o que um relatório limpo consegue e não consegue provar.
3. Configurar K-fold de classificação com o CSV, salvar a configuração e conferir
   que o caminho foi preservado. Não exigir treinamento de todos os folds.
4. Executar um treino curto de classificação e localizar o resultado no histórico.
5. Exportar o ZIP desse run e identificar o comando que verifica sua integridade.
6. Tentar um caminho inválido e observar se a mensagem permite corrigir o pedido.

## Registro e encerramento

Para cada tarefa, anotar resultado, tempo observado, ajuda e ponto de confusão;
não substituir falhas por tentativas bem-sucedidas posteriores. Perguntar qual
informação faltou para decidir o próximo passo. Separar falha de produto, falha
de instalação e dificuldade de compreensão com evidência da sessão.

Corrigir os bloqueios antes de congelar a versão da coleta válida. Se as tarefas
ou hipóteses do estudo mudarem, registrar a revisão e seus motivos antes da
coleta. Não atribuir ao software o efeito de ajuda prestada pelo facilitador.
