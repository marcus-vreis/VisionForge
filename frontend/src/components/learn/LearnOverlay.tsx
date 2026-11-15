/** "Começando do zero": the full-screen guide for someone who has never trained
 * a model (ADR-207).
 *
 * The two tours point at the screen; this one explains the field behind it, so
 * it gets a page of its own instead of a card: chapters read top to bottom, a
 * rail on the left says where the reader is, and each chapter has a figure. It
 * opens from the header's guide menu and from the first-visit invitation, and
 * only there: nothing here opens by itself.
 *
 * The chapter list and every link live in lib/learn.ts; the words live in the
 * dictionaries (`learn`).
 */

import { useEffect, useRef, useState, type ReactNode } from "react";

import dataStepUrl from "../../assets/learn/data-step.webp";
import resultsUrl from "../../assets/learn/results.webp";
import filmUrl from "../../assets/learn/training-film.webp";
import { useT } from "../../i18n/useT";
import type { Dict } from "../../i18n/pt";
import {
  LEARN_CHAPTERS,
  PROJECT_DOCS,
  RESOURCES,
  STACK,
  chapterAnchor,
  type LearnChapterId,
} from "../../lib/learn";
import {
  BatchesFigure,
  ConfusionFigure,
  CurvesFigure,
  ScreenFigure,
  SplitFigure,
  TrainingLoopFigure,
} from "./LearnFigures";

type Chapters = Dict["learn"]["chapters"];

/** The size of the captured screens (scripts/capture_docs_images.py). */
const STILL = { width: 1100, height: 733 };
const FILM = { width: 960, height: 640 };

function Terms({ label, terms }: { label: string; terms: readonly (readonly string[])[] }) {
  return (
    <div className="vf-learn-terms">
      <div className="vf-learn-terms-label">{label}</div>
      <dl>
        {terms.map(([term, meaning]) => (
          <div key={term}>
            <dt>{term}</dt>
            <dd>{meaning}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** A link that leaves the app: always a new tab, never the opener. */
function OutLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a className="vf-learn-link" href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  );
}

function LinkList({
  items,
  hint,
}: {
  items: readonly { key: string; name: string; url: string; text?: string }[];
  hint: string;
}) {
  return (
    <ul
      className={
        items.some((item) => item.text) ? "vf-learn-links" : "vf-learn-links vf-learn-links--plain"
      }
    >
      {items.map((item) => (
        <li key={item.key}>
          <OutLink href={item.url}>
            {item.name}
            <span aria-hidden="true"> ↗</span>
            <span className="vf-learn-sr"> ({hint})</span>
          </OutLink>
          {item.text && <span>{item.text}</span>}
        </li>
      ))}
    </ul>
  );
}

/** What a chapter shows under its paragraphs. */
function ChapterExtras({
  id,
  chapters,
  onFirstTraining,
}: {
  id: LearnChapterId;
  chapters: Chapters;
  onFirstTraining: () => void;
}) {
  const t = useT().learn;
  switch (id) {
    case "idea":
      return <TrainingLoopFigure />;
    case "data":
      return (
        <>
          <SplitFigure />
          <ScreenFigure
            src={dataStepUrl}
            alt={chapters.data.screenAlt}
            caption={chapters.data.screenCaption}
            {...STILL}
          />
        </>
      );
    case "loop":
      return <BatchesFigure />;
    case "curves":
      return <CurvesFigure />;
    case "result":
      return (
        <>
          <ConfusionFigure />
          <ScreenFigure
            src={resultsUrl}
            alt={chapters.result.screenAlt}
            caption={chapters.result.screenCaption}
            {...STILL}
          />
        </>
      );
    case "watch":
      return (
        <>
          <ScreenFigure
            src={filmUrl}
            alt={chapters.watch.filmAlt}
            caption={chapters.watch.filmCaption}
            {...FILM}
          />
          <button type="button" className="vf-learn-cta" onClick={onFirstTraining}>
            {chapters.watch.start}
            <span aria-hidden="true"> →</span>
          </button>
        </>
      );
    case "stack":
      return (
        <LinkList
          hint={t.opensOutside}
          items={STACK.map((lib) => ({
            key: lib.id,
            name: lib.name,
            url: lib.url,
            text: chapters.stack.items[lib.id],
          }))}
        />
      );
    case "more":
      return (
        <>
          <h3 className="vf-learn-subtitle">{chapters.more.resourcesTitle}</h3>
          <LinkList
            hint={t.opensOutside}
            items={RESOURCES.map((r) => ({
              key: r.id,
              name: r.name,
              url: r.url,
              text: chapters.more.resources[r.id],
            }))}
          />
          <h3 className="vf-learn-subtitle">{chapters.more.docsTitle}</h3>
          <LinkList
            hint={t.opensOutside}
            items={PROJECT_DOCS.map((d) => ({
              key: d.id,
              name: chapters.more.docs[d.id],
              url: d.url,
            }))}
          />
          <p className="vf-learn-footnote">{chapters.more.footnote}</p>
        </>
      );
  }
}

function termsOf(id: LearnChapterId, chapters: Chapters): readonly (readonly string[])[] | null {
  const chapter = chapters[id];
  return "terms" in chapter ? chapter.terms : null;
}

/** The chapter whose heading is nearest above the top of the scroller. */
function useActiveChapter(scroller: React.RefObject<HTMLDivElement | null>): LearnChapterId {
  const [active, setActive] = useState<LearnChapterId>(LEARN_CHAPTERS[0]);
  useEffect(() => {
    const node = scroller.current;
    if (!node) return;
    const onScroll = () => {
      const top = node.getBoundingClientRect().top;
      let current: LearnChapterId = LEARN_CHAPTERS[0];
      for (const id of LEARN_CHAPTERS) {
        const section = document.getElementById(chapterAnchor(id));
        if (section && section.getBoundingClientRect().top - top <= 120) current = id;
      }
      setActive(current);
    };
    onScroll();
    node.addEventListener("scroll", onScroll, { passive: true });
    return () => node.removeEventListener("scroll", onScroll);
  }, [scroller]);
  return active;
}

export function LearnOverlay({
  onClose,
  onFirstTraining,
}: {
  onClose: () => void;
  /** Closes this screen and starts the "Primeiro treino" guide. */
  onFirstTraining: () => void;
}) {
  const t = useT().learn;
  const scroller = useRef<HTMLDivElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const active = useActiveChapter(scroller);

  useEffect(() => {
    closeButton.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  // On a narrow screen the rail is a strip that scrolls sideways: keep the
  // current chapter in sight.
  useEffect(() => {
    document
      .querySelector('.vf-learn-railitem[aria-current="true"]')
      ?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [active]);

  const goTo = (id: LearnChapterId) => {
    document.getElementById(chapterAnchor(id))?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div
      className="vf-learn"
      role="dialog"
      aria-modal="true"
      aria-labelledby="learn-title"
      data-testid="learn-overlay"
    >
      <nav className="vf-learn-rail" aria-label={t.railLabel}>
        <div className="vf-learn-eyebrow">{t.eyebrow}</div>
        <ol>
          {LEARN_CHAPTERS.map((id, i) => (
            <li key={id}>
              <button
                type="button"
                className="vf-learn-railitem"
                aria-current={id === active ? "true" : undefined}
                onClick={() => goTo(id)}
              >
                <span>{String(i + 1).padStart(2, "0")}</span>
                {t.chapters[id].title}
              </button>
            </li>
          ))}
        </ol>
      </nav>

      <div className="vf-learn-scroll" ref={scroller}>
        <button
          ref={closeButton}
          type="button"
          className="vf-learn-close"
          onClick={onClose}
          title={t.closeTitle}
        >
          <span aria-hidden="true">✕</span> {t.close}
        </button>

        <header className="vf-learn-hero">
          <div className="vf-learn-eyebrow">{t.eyebrow}</div>
          <h1 id="learn-title">{t.title}</h1>
          <p>{t.lead}</p>
        </header>

        {LEARN_CHAPTERS.map((id, i) => {
          const chapter = t.chapters[id];
          const terms = termsOf(id, t.chapters);
          // The curves chapter asks the reader to try the figure, so the figure
          // comes right after that paragraph; elsewhere it follows the text.
          const lead = id === "curves" ? 1 : chapter.body.length;
          return (
            <section key={id} id={chapterAnchor(id)} className="vf-learn-chapter" data-chapter={id}>
              <div className="vf-learn-number">
                {t.chapterOf(i + 1, LEARN_CHAPTERS.length)}
              </div>
              <h2>{chapter.title}</h2>
              {chapter.body.slice(0, lead).map((paragraph) => (
                <p key={paragraph}>{paragraph}</p>
              ))}
              <ChapterExtras id={id} chapters={t.chapters} onFirstTraining={onFirstTraining} />
              {chapter.body.slice(lead).map((paragraph) => (
                <p key={paragraph}>{paragraph}</p>
              ))}
              {terms && <Terms label={t.termsLabel} terms={terms} />}
            </section>
          );
        })}
      </div>
    </div>
  );
}
