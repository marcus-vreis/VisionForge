import { Component, type ErrorInfo, type ReactNode } from "react";
import { useT } from "../i18n/useT";

interface BoundaryProps {
  children: ReactNode;
  /** What stands in for the children once one of them throws while rendering. */
  fallback: ReactNode;
  /** When this changes, a boundary that has tripped tries its children again. */
  resetKey?: string;
}

interface BoundaryState {
  failed: boolean;
  /** The `resetKey` the current state was reached under. */
  resetKey: string | undefined;
}

/**
 * Keeps one throwing component from taking the whole page with it.
 *
 * React unmounts the entire tree when a render throws and nothing catches it,
 * which reads as a blank page. This is the catch; the cause still has to be
 * fixed where it starts (a bad value let into the form), the boundary only
 * decides what the person sees meanwhile.
 */
export class ErrorBoundary extends Component<BoundaryProps, BoundaryState> {
  state: BoundaryState = { failed: false, resetKey: this.props.resetKey };

  static getDerivedStateFromError(): Partial<BoundaryState> {
    return { failed: true };
  }

  static getDerivedStateFromProps(
    props: BoundaryProps,
    state: BoundaryState,
  ): Partial<BoundaryState> | null {
    // Moving to another task tab is a fresh start for what the boundary guards.
    return props.resetKey === state.resetKey
      ? null
      : { failed: false, resetKey: props.resetKey };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // The notice is deliberately short; the detail goes where a bug report can find it.
    console.error(error, info.componentStack);
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

const noticeStyle: React.CSSProperties = {
  padding: 28,
  background: "var(--vf-panel)",
  border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
  borderRadius: 18,
  backdropFilter: "blur(14px)",
  display: "flex",
  flexDirection: "column",
  gap: 10,
  alignItems: "flex-start",
};

const kickerStyle: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.22em",
  textTransform: "uppercase",
  color: "oklch(0.85 0.14 22)",
};

const reloadStyle: React.CSSProperties = {
  marginTop: 6,
  padding: "8px 14px",
  background: "var(--accent-soft)",
  border: "1px solid var(--accent-vf)",
  borderRadius: 10,
  color: "var(--vf-text)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.10em",
  textTransform: "uppercase",
  cursor: "pointer",
  lineHeight: 1,
};

/** The short message and the way out that a tripped boundary shows. */
export function CrashNotice({ onReload }: { onReload: () => void }) {
  const t = useT();
  return (
    <div role="alert" style={noticeStyle}>
      <div style={kickerStyle}>{t.errorBoundary.title}</div>
      <div style={{ color: "var(--vf-text-dim)", fontSize: 14, lineHeight: 1.55 }}>
        {t.errorBoundary.body}
      </div>
      <button type="button" onClick={onReload} style={reloadStyle}>
        {t.errorBoundary.reload}
      </button>
    </div>
  );
}

/** The boundary the app puts around its main content: a crash there leaves the
 *  header, the tabs and the bottom bar standing, with the notice in its place. */
export function ContentBoundary({
  children,
  resetKey,
}: {
  children: ReactNode;
  resetKey?: string;
}) {
  return (
    <ErrorBoundary
      resetKey={resetKey}
      fallback={<CrashNotice onReload={() => window.location.reload()} />}
    >
      {children}
    </ErrorBoundary>
  );
}
