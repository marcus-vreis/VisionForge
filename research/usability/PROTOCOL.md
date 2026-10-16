# First-tester evaluation — protocol

Registered by the commit that adds this file. Any change after it is recorded
under **Deviations**, with the date and the reason.

## Purpose

Two questions, measured in one session per participant:

1. **Usability.** Can someone who has never seen VisionForge install it from the
   README and complete the core workflow unaided, and how usable do they find it
   (System Usability Scale)?
2. **Learning.** Does the first-training guide, with the "Como funciona" tab's
   animations, improve a beginner's understanding of how training works? Measured
   as the change in a short concept quiz answered before and after the session —
   not as self-report, which is weak evidence on its own.

## Participants

The first 5–15 people invited to test: students and researchers who train, or
want to train, vision models. No name, e-mail or other identifying data is
collected (e-mail collection off in the form). The only background questions are
coarse (experience level, operating system, GPU vendor), so no answer identifies
anyone.

Ethics: Resolução CNS 510/2016 exempts anonymous opinion surveys from CEP review;
confirm with the advisor or the institution's CEP before reporting results in a
paper. This is not legal advice.

## Consent (first page of the form)

See `FORM.md`, section 0. Continuing means agreeing; anyone can stop at any time.

## Procedure

1. Background (3 questions) and the **pre-quiz** (5 questions), before opening
   VisionForge.
2. Tasks, timed by the participant (start and end clock time each):
   1. Install VisionForge following only the README, and open the interface.
   2. Open the "Primeiro treino" guide and follow it to the end with its example
      dataset.
   3. While a training runs, open the "Como funciona" tab and watch it.
   4. Run the same configuration with 3 seeds ("Rodar réplicas") and read the
      mean ± 95% CI.
   5. In the history, open a run and download its model card.

   Per task: completed / completed with help / not completed; start and end time;
   what went wrong, if anything (free text).
3. **Post-quiz**: the same 5 questions as the pre-quiz, in a different order.
4. SUS (10 items) and two open questions.

Participants receive only the PyPI link and the README: task 1 measures whether
that is enough.

## Instruments

- **SUS** (Brooke, 1996), in the validated Brazilian-Portuguese version of
  Lourenço, Carmona & Lopes (2022), *Aquichan* 22(2), e2228,
  doi:10.5294/aqui.2022.22.2.8. Item 7 follows the consensus wording reported in
  their Results ("… muito rapidamente"), which matches the original "very
  quickly". Answers 1 (discordo totalmente) to 5 (concordo totalmente).
- **Concept quiz**: 5 multiple-choice questions, one correct option each, on the
  ideas the guide and the animations teach (validation split, epoch, overfitting
  in the curves, early stopping, why repeat with other seeds). Written for this
  study; not a validated instrument, and reported as such. Q4 (early stopping)
  is not explained by the guide or the animations as of 0.15.0 — the interface
  only names it — so it is read separately, as a rough check on how much any gain
  comes from using the software rather than from the guide.
- **Open questions**: "Qual foi o momento mais confuso?" and "O que faria você
  usar isto na sua própria pesquisa?".

## Analysis

- SUS per participant (`sus.py`), then mean, SD and range.
- Task completion rate per task; median time per task.
- Quiz: score 0–5 before and after, per participant; the paired gain
  (post − pre), with its mean, and how many participants improved, stayed or got
  worse.

With n ≤ 15 everything is descriptive: means are reported with their range, not
as population estimates, and no significance test is claimed. The quiz gain has
no control group, so it cannot separate the guide's effect from simply using the
software for an hour; this limitation is stated wherever the gain is reported.

## Deviations

None yet.
