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
2. Tasks, timed by the facilitator with a stopwatch, in seconds (see
   **Facilitation and timing**):
   1. Install VisionForge following only the README (and `docs/INSTALL.md`, which it
      links for each system), and open the interface.
   2. Open the "Primeiro treino" guide and follow it to the end with its example
      dataset.
   3. While a training runs, open the "Como funciona" tab and watch it.
   4. Run the same configuration with 3 seeds ("Rodar réplicas") and read the
      mean ± 95% CI.
   5. In the history, open a run and download its model card.

   Per task, the facilitator records the outcome, the elapsed seconds and the
   number of interventions (`session_template.csv`); the participant answers,
   in the form, how it went and what went wrong, if anything (free text).
3. **Post-quiz**: the same 5 questions as the pre-quiz, in a different order.
4. SUS (10 items) and two open questions.

Participants receive only the PyPI link, the README and `docs/INSTALL.md` (added on
2026-10-19, before the first participant): task 1 measures whether that is enough.

## Facilitation and timing

- **Facilitator.** One person runs every session, with the same opening script
  and the same answer to "how do I…?": *"Faça como faria sozinho(a); se travar,
  me diga e eu ajudo."* Who facilitates is decided with the advisor (see
  **Pending decisions**); while the facilitator is the author, that is a stated
  limitation.
- **Intervention.** Any hint, answer, demonstration or action by the facilitator
  that concerns how to do the task. Reading the task statement again, word for
  word, and fixing the recording or the session itself are not interventions.
  Each one is counted in `assistance_count`.
- **Outcome**, one per task, from the facilitator's record:
  `completed` (finished with 0 interventions), `assisted` (finished with 1 or
  more), `failed` (not finished, or stopped by the participant, after starting),
  `withdrawn` (the participant left the study before the task). The form's
  "Consegui com ajuda" is kept as the participant's view, but the analysis uses
  the facilitator's `outcome`.
- **Time.** `elapsed_seconds` from the moment the participant has read the task
  to the moment it is done or abandoned, on a stopwatch. Tasks 3 and 5 take
  under a minute, so clock times in minutes would round them to 0 or 1; the
  clock times in the form are only a cross-check.
- During the post-quiz and the SUS the facilitator stays out of the
  participant's view and does not comment on answers.

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
as population estimates, and no significance test is claimed.

- **SUS** is reported as each participant's score, plus mean, SD and range. No
  adjective scale or percentile is attached, and the SUS is never read as a
  task success rate.
- **Times** are reported as median and range per task, in seconds, over the
  participants who completed or were assisted; failures are counted, not timed.

## Limitations

Stated wherever the corresponding result is reported:

- **No control group.** The quiz gain cannot separate the guide's effect from
  simply using the software for an hour.
- **Practice effect.** The post-quiz repeats the pre-quiz questions in the same
  session, minutes later, only reordered. Part of any gain can come from having
  seen the questions; the gain is read as an upper bound.
- **Facilitator is the author.** While the person who built VisionForge
  facilitates, participants may rate it more kindly (SUS) and try harder; the
  facilitator also decides what counts as an intervention. The fixed script, the
  definition above and staying out of view during the SUS reduce this, but do not
  remove it.
- **Small, invited sample.** 5–15 people invited by the author are not a random
  sample of the target users.
- **Concept quiz not validated** (see **Instruments**).

## Pending decisions

To settle with the advisor before the first participant, then record here:

- Who facilitates (the author, or someone else with the same script).
- Target n within 5–15, and the stopping rule.
- Whether the protocol is pre-registered externally (e.g. OSF) before the first
  session.

## Deviations

- **2026-11-12, before the first participant** (#229): timing moved from the
  participant's clock times (minute resolution) to the facilitator's stopwatch
  in seconds; "completed with help" defined as 1 or more counted interventions,
  the same field in the form, the runbook and the session sheet; the practice
  effect and the facilitator being the author added as limitations; SUS and time
  reporting written down; open decisions listed under **Pending decisions**.
