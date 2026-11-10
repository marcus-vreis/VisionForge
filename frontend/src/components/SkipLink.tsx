import { useT } from "../i18n/useT";

/** First focusable element of the page: Tab jumps the header and tabs to the content. */
export function SkipLink() {
  const t = useT();
  return (
    <a className="vf-skip-link" href="#vf-main">
      {t.a11y.skipToContent}
    </a>
  );
}
