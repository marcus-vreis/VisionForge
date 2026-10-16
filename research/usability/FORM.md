# Formulário — texto para colar no Google Forms

Um formulário só, com seções nesta ordem: as respostas de antes e de depois ficam
na mesma resposta, o que pareia o quiz sem pedir nenhum identificador.

Configurações do Google Forms:

- **Configurações → Respostas → Coletar endereços de e-mail: Não coletar.**
- Desligar "Limitar a 1 resposta" (exigiria login).
- Não ativar o modo teste (quiz): a nota não deve aparecer para o participante
  no meio do formulário, senão o quiz de depois vira cola do de antes. O gabarito
  fica neste arquivo e a pontuação é feita na análise.
- Avisar o participante para manter a aba aberta durante a sessão: sem login, o
  Google Forms não guarda o rascunho.

---

## Seção 0 — Consentimento

> Este é um teste voluntário e anônimo do VisionForge, um software de visão
> computacional. Você vai responder algumas perguntas, usar o software por cerca de
> uma hora seguindo algumas tarefas e responder um questionário curto. Não
> coletamos nome, e-mail nem nenhum dado que identifique você, e você pode parar a
> qualquer momento. As respostas podem ser publicadas de forma agregada em
> trabalhos sobre o software. Continuar significa que você concorda.

Pergunta (múltipla escolha, obrigatória): **Você concorda em participar?**
— Sim, concordo / Não concordo (a opção "Não concordo" envia o formulário).

## Seção 1 — Sobre você

1. **Qual sua experiência com treino de redes neurais?** — Nunca treinei / Já
   segui um tutorial / Treino de vez em quando / Treino com frequência
2. **Sistema operacional que vai usar:** — Windows / macOS (Apple Silicon: M1, M2…)
   / macOS (Intel) / Linux
3. **Placa de vídeo:** — NVIDIA / AMD / Apple (chip M) / Nenhuma ou não sei

## Seção 2 — Antes de começar (5 perguntas)

Instrução: *Responda sem pesquisar. "Não sei" é uma resposta válida e ajuda o
estudo.*

Use as perguntas Q1–Q5 do bloco "Quiz" abaixo, nesta ordem: Q1, Q2, Q3, Q4, Q5.

## Seção 3 — Tarefas

Instrução: *Você recebeu só o link do PyPI e o README. Para cada tarefa, anote a
hora em que começou e a hora em que terminou (ex.: 14:05).*

Para cada tarefa T1–T5, quatro perguntas:

- Hora de início (resposta curta)
- Hora de fim (resposta curta)
- Resultado — Consegui sozinho(a) / Consegui com ajuda / Não consegui
- O que deu errado ou confundiu, se algo? (parágrafo, opcional)

Tarefas:

- **T1.** Instale o VisionForge seguindo só o README e abra a interface.
- **T2.** Abra o guia "Primeiro treino" (botão "guia" no cabeçalho) e siga até o
  fim, com o dataset de exemplo que ele oferece.
- **T3.** Enquanto um treino roda, abra a aba "Como funciona" e acompanhe.
- **T4.** Rode a mesma configuração com 3 seeds ("Rodar réplicas") e leia a média
  ± IC 95%. Qual foi a média? (resposta curta)
- **T5.** No histórico, abra um run e baixe o model card.

## Seção 4 — Depois das tarefas (as mesmas 5 perguntas)

Instrução: *As mesmas perguntas do começo. Responda sem voltar às respostas
anteriores.*

Ordem: Q3, Q5, Q1, Q4, Q2.

## Seção 5 — Questionário de usabilidade (SUS)

Instrução: *Para cada frase, marque de 1 (discordo totalmente) a 5 (concordo
totalmente). "Sistema" é o VisionForge.*

Escala linear 1–5, rótulos "Discordo totalmente" e "Concordo totalmente".
Versão brasileira validada de Lourenço, Carmona & Lopes (2022), *Aquichan*
22(2), e2228, doi:10.5294/aqui.2022.22.2.8. **Não alterar a redação nem a ordem.**

1. Eu acho que gostaria de usar esse sistema frequentemente.
2. Eu achei esse sistema desnecessariamente complexo.
3. Eu achei esse sistema fácil de usar.
4. Eu achei que precisaria de ajuda de uma pessoa técnica para ser capaz de usar esse sistema.
5. Eu achei que as várias funções desse sistema foram bem integradas.
6. Eu acho que o sistema apresenta muita inconsistência.
7. Eu imagino que a maioria das pessoas pode aprender a usar esse sistema muito rapidamente.
8. Eu achei esse sistema muito pesado para usar.
9. Eu me senti muito seguro usando o sistema.
10. Eu precisei aprender muitas coisas antes que pudesse utilizar esse sistema.

## Seção 6 — Perguntas abertas

1. Qual foi o momento mais confuso?
2. O que faria você usar isto na sua própria pesquisa?

Lembrete no fim: *Não escreva nome, e-mail nem nada que identifique você.*

---

## Quiz (Q1–Q5) — múltipla escolha, uma resposta, com "Não sei"

Embaralhar as opções desligado (a ordem fixa facilita a análise). Gabarito
marcado com ✔ — **não aparece no formulário**.

**Q1. Para que serve o conjunto de validação?**
- a) Dar mais imagens para o modelo treinar.
- b) Acompanhar o treino e decidir quando parar ou qual configuração usar, sem tocar no conjunto de teste. ✔
- c) Medir a nota final que vai para o relatório.
- d) Não sei.

**Q2. O que é uma época?**
- a) Uma passada do modelo por todas as imagens de treino. ✔
- b) Um lote (batch) de imagens.
- c) O tempo total do treino.
- d) Não sei.

**Q3. No gráfico, a loss de treino continua caindo, mas a de validação sobe por várias épocas seguidas. O que isso indica mais provavelmente?**
- a) O modelo está aprendendo cada vez melhor.
- b) O modelo está decorando as imagens de treino (overfitting). ✔
- c) A taxa de aprendizado é baixa demais.
- d) Não sei.

**Q4. O que faz o early stopping?**
- a) Para o treino quando a loss de validação deixa de melhorar por algumas épocas. ✔
- b) Para o treino quando a acurácia de treino chega a 100%.
- c) Reduz o número de imagens para treinar mais rápido.
- d) Não sei.

**Q5. A configuração A deu 85% de acurácia e a B deu 83%, cada uma treinada uma vez só. O que dá para concluir?**
- a) A é melhor que B.
- b) B é melhor, porque 2 pontos é pouco.
- c) Pouco: a diferença pode ser só sorte da seed; é preciso repetir com várias seeds. ✔
- d) Não sei.

Pontuação: 1 por acerto; "Não sei" e erro valem 0. Nota de 0 a 5, antes e depois.
