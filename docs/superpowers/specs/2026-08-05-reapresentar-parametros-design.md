# Reapresentar os parâmetros

**Data:** 2026-08-05
**Estado:** desenhado, aprovado, não implementado
**Escopo:** sub-projeto (d) de quatro.

## Problema

Relato do usuário: *"conheço apenas metade desses. Tá bem difícil de entender,
muita coisa junta."*

E, explicitamente: **não se trata de eliminar parâmetros**. O problema é como
estão apresentados, não quantos são.

## Decisões

### 1. Revelação progressiva, reusando a chave de (b)

Cada seção de hiperparâmetros ganha uma divisão **Básico / Avançado**. O básico
fica sempre visível; o avançado é colapsado por padrão, com o mesmo padrão de
esconder que (b) estabelece — inclusive o contador de quantos campos estão
guardados, para o painel não parecer quebrado.

Nada é removido. Um parâmetro em "Avançado" está a um clique de distância e
continua no payload com o mesmo valor de sempre.

**Básico** (sempre visível): épocas, batch size, learning rate, seed.

**Avançado** (colapsado): otimizador e seus parâmetros, scheduler e seus
dependentes, early stopping, AMP, `num_workers`, `pin_memory`, regularização.

O corte não é por importância teórica — otimizador e scheduler importam muito.
É por **frequência de ajuste**: os quatro básicos são os que mudam entre um
experimento e o seguinte; o resto se define uma vez e fica.

### 2. Uma linha explicando cada parâmetro

Hoje existe rótulo, não existe explicação. Cada campo ganha uma linha curta
dizendo o que o parâmetro faz e o efeito de aumentá-lo ou diminuí-lo — não a
definição de livro-texto, a consequência prática.

Isso é o que ataca diretamente o "conheço metade": um parâmetro desconhecido com
uma linha de explicação deixa de ser desconhecido, e nenhum parâmetro precisou
sair da tela para isso.

### 3. `num_workers` e `pin_memory` ganham a advertência do WinError 1455

`num_workers` é o parâmetro que derrubou um treino real (ADR-081): cada worker é
um processo que recarrega o torch e as DLLs da CUDA, ~1 GB cada, multiplicado
pelo número de loaders. A linha de explicação dele diz isso, porque é o único
parâmetro da lista cujo valor errado não degrada o treino — impede o treino.

## Fora de escopo

- Busca de parâmetros no painel. Faz sentido quando houver muito mais campos;
  hoje resolveria menos que a divisão básico/avançado.
- Presets de hiperparâmetros por tarefa. Decisão separada.
- Remover qualquer parâmetro. Explicitamente recusado pelo usuário.

## Testes

- O round-trip `formFromPayload(buildPayload(form)) == form` continua valendo com
  a seção avançada colapsada — colapsar é apresentação, não muda o payload.
- Um valor não-default num campo avançado força a seção a abrir: esconder um
  ajuste deliberado seria pior que a poluição visual que motivou a mudança.
- Todo campo renderizado tem texto de explicação (teste de completude sobre o
  mapa de rótulos, para nenhum campo novo entrar sem explicação).
