# SVG animations — notes & direction

> Status: **idea / direction only.** No work scheduled. Captures where animated
> SVG fits VisionForge and the guardrails so it stays an asset, not noise.

## Where we are today

The UI already uses animated SVG: `Waves.tsx` (a moving wave field) plus
`Particles`, rendered as the page background. App.tsx deliberately isolates them
in their own compositor layer (`transform: translateZ(0)`, `contain: layout
paint`) so the `backdrop-filter` glass panels above don't drag the background
through a repaint on every scroll. That isolation is the pattern to keep.

## Where more animation would earn its place

Tasteful, low-frequency, meaning-bearing motion — not decoration:
- **Empty / loading states** — the history "nenhum treinamento" and the training
  overlay's pre-epoch crawl are the natural homes for a small animated glyph.
- **Metric sparklines** — a tiny inline SVG trend on each history `RunCard`
  (loss or mAP over epochs) would add signal, not just polish.
- **Task-accent transitions** — the per-task accent already shifts on tab change;
  an SVG accent flourish could reinforce it.
- **Live training** — a subtle progress-tied SVG in `TrainingOverlay` keyed to the
  real epoch fraction (it already computes one).

## Guardrails
- **Performance first.** Keep animated SVG off the hot path; reuse the isolated
  compositor-layer pattern. No animation that forces layout/paint on scroll.
- **Respect `prefers-reduced-motion`** — gate non-essential motion behind it.
- **Motion must mean something** — tie it to state (progress, a metric, a
  transition), never animate for its own sake.
- **No new heavy dependency** — hand-authored SVG + CSS/`requestAnimationFrame`,
  as today. A framer-motion-scale dependency is not justified for this.
