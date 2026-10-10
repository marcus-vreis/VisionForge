import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useT } from "../i18n/useT";
import {
  GUIDES,
  guideById,
  type ActionResult,
  type GuideContext,
  type GuideId,
  type GuideStep,
} from "../lib/guides";
import { gateOpen } from "../lib/guides/gates";
import { FLOATING_WIDTH, placeFloating } from "../lib/guides/placement";
import { useGuideFacts } from "../lib/guides/useGuideFacts";
import { CARD_WIDTH, markTourSeen, placeCard } from "../lib/tour";

/** Os guias (ADR-104, ADR-115): o tour da interface e o primeiro treino.
 *
 * Um passo pode esperar o que o pesquisador faz (`waitFor`, lido do estado real
 * da tela), oferecer um botão que faz o passo (`action`) e pedir um cartão que
 * não escurece a página (`floating`), para os passos em que ele precisa olhar a
 * tela de trás. Os passos do tour não têm nada disso e se comportam como antes.
 *
 * Um recorte de luz sobre o elemento de que o passo está falando e um cartão ao
 * lado dele. A escuridão são quatro painéis ao redor do recorte, cada um com a
 * sua transição, e não uma sombra com espalhamento gigante na caixa do recorte:
 * a sombra é um jeito mais curto de escrever a mesma coisa, mas obriga o
 * navegador a rasterizar uma camada de ~20000px de lado a cada quadro da
 * animação, e nessa página ela chegou a segurar a pintura por segundos. Os
 * quatro painéis interpolam igual e custam o de sempre.
 *
 * O recorte não recebe cliques (`pointer-events: none`) — quem quiser ignorar o
 * guia e mexer na interface por baixo consegue, e o ✕, o "pular" e o Esc estão
 * sempre à mão. Um passo cujo alvo não existe na tarefa ativa não some: vira um
 * cartão centralizado, o que mantém um roteiro só para as cinco tarefas.
 */

const EASE = "cubic-bezier(.16,.84,.24,1)";
const MOVE = ["left", "top", "right", "bottom", "width", "height"]
  .map((prop) => `${prop} 460ms ${EASE}`)
  .join(", ");

interface GuidedTourProps {
  /** Começa pelo convite ("quer um guia?") em vez de já entrar no primeiro passo. */
  invite?: boolean;
  /** O guia a tocar quando não há convite. */
  guide?: GuideId;
  /** Os eventos do treino em curso; os portões "começou" e "terminou" leem daqui. */
  events: readonly { event: string }[];
  /** A pasta do dataset do formulário de classificação. */
  datasetPath: string;
  /** O que um passo pode mudar na aplicação. */
  context: GuideContext;
  onClose: () => void;
}

export function GuidedTour({
  invite = false,
  guide = "tour",
  events,
  datasetPath,
  context,
  onClose,
}: GuidedTourProps) {
  const t = useT();
  const [guideId, setGuideId] = useState<GuideId | null>(invite ? null : guide);
  const steps = useMemo(
    () => (guideId ? guideById(guideId).steps(t) : []),
    [guideId, t],
  );
  const [step, setStep] = useState(invite ? -1 : 0);
  // O que já estava na lista de eventos quando o guia abriu não conta: a trava
  // só abre um portão por um evento que aparece depois (lib/guides/gates).
  const facts = useGuideFacts(events, datasetPath);
  const [action, setAction] = useState<{
    step: number;
    busy: boolean;
    result: ActionResult | null;
  } | null>(null);
  const contextRef = useRef(context);
  useEffect(() => {
    contextRef.current = context;
  }, [context]);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const [animate, setAnimate] = useState(true);
  const [entered, setEntered] = useState(false);
  // A altura do cartão muda a cada passo (os textos têm tamanhos diferentes) e
  // decide se ele cabe abaixo do alvo. Observá-la é mais direto do que medir
  // depois de pintar e reposicionar num segundo quadro.
  const [cardHeight, setCardHeight] = useState(220);
  const observer = useRef<ResizeObserver | null>(null);
  const cardRef = useCallback((node: HTMLDivElement | null) => {
    observer.current?.disconnect();
    if (!node) return;
    const ro = new ResizeObserver(() => setCardHeight(node.offsetHeight));
    ro.observe(node);
    observer.current = ro;
  }, []);

  const current = step >= 0 ? (steps[step] ?? null) : null;
  const anchor = current?.anchor;
  const floating = current?.floating === true;
  const alignTop = current?.align === "top";
  const last = step === steps.length - 1;
  const open = current ? gateOpen(current, facts) : true;
  const shownAction = action && action.step === step ? action : null;

  const finish = useCallback(() => {
    markTourSeen();
    onClose();
  }, [onClose]);

  const choose = useCallback((id: GuideId) => {
    setGuideId(id);
    setStep(0);
  }, []);

  const runAction = useCallback(async () => {
    const step_action = current?.action;
    if (!step_action) return;
    const at = step;
    setAction({ step: at, busy: true, result: null });
    const result = await step_action.run(contextRef.current);
    setAction({ step: at, busy: false, result });
  }, [current, step]);

  // Ao abrir um passo, o que ele pede à aplicação (trocar de aba, guardar a tela
  // de treino). Depende só do passo: ler o contexto por ref evita repetir o
  // efeito a cada render do App.
  const onEnter = current?.onEnter;
  useEffect(() => {
    onEnter?.(contextRef.current);
  }, [onEnter]);

  const measure = useCallback(() => {
    if (!anchor) {
      setRect(null);
      return;
    }
    const el = document.querySelector(`[data-tour="${anchor}"]`);
    setRect(el ? el.getBoundingClientRect() : null);
  }, [anchor]);

  // Entrada: um quadro para o fade pegar, já que o overlay monta opaco.
  useEffect(() => {
    const id = window.requestAnimationFrame(() => setEntered(true));
    return () => window.cancelAnimationFrame(id);
  }, []);

  // A cada passo: traz o alvo para a tela e mede duas vezes — uma no quadro
  // seguinte, para o foco já sair do lugar, e outra depois que a rolagem suave
  // terminou, que é quando a posição final finalmente vale.
  useEffect(() => {
    const el = anchor
      ? document.querySelector(`[data-tour="${anchor}"]`)
      : null;
    // A tela de treino é fixa e é o que o passo flutuante quer ver inteira.
    if (el && !floating) {
      if (alignTop) {
        // Um passo de cartão alto (o botão do dataset de exemplo) não cabe nem
        // acima nem abaixo de um alvo no meio da tela, e centralizado cobriria o
        // campo que o passo mostra. Com o alvo perto do topo, o cartão tem o
        // resto da tela abaixo dele.
        window.scrollTo({
          top: Math.max(0, el.getBoundingClientRect().top + window.scrollY - 48),
          behavior: "smooth",
        });
      } else {
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
    const raf = window.requestAnimationFrame(() => {
      setAnimate(true);
      measure();
    });
    const id = window.setTimeout(measure, 380);
    return () => {
      window.cancelAnimationFrame(raf);
      window.clearTimeout(id);
    };
  }, [anchor, floating, alignTop, measure]);

  // Rolagem e redimensionamento acompanham sem transição: interpolar aqui faria
  // o recorte perseguir a página com atraso em vez de ficar colado nela.
  useEffect(() => {
    const track = () => {
      setAnimate(false);
      measure();
    };
    window.addEventListener("scroll", track, true);
    window.addEventListener("resize", track);
    // Aba em segundo plano congela o requestAnimationFrame que faz a medida do
    // passo, então quem volta para a aba pode encontrar o foco no alvo antigo.
    document.addEventListener("visibilitychange", track);
    return () => {
      window.removeEventListener("scroll", track, true);
      window.removeEventListener("resize", track);
      document.removeEventListener("visibilitychange", track);
    };
  }, [measure]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        finish();
      } else if (e.key === "ArrowRight") {
        // Um portão fechado vale para a seta como para o botão.
        if (open) setStep((s) => (s + 1 >= steps.length ? s : s + 1));
      } else if (e.key === "ArrowLeft") {
        setStep((s) => (s > 0 ? s - 1 : s));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [finish, steps.length, open]);

  // O cartão vai abaixo do alvo, ou acima quando não sobra espaço; o convite e
  // os passos sem alvo ficam no centro. O passo flutuante fica ao lado do alvo.
  const view = { width: window.innerWidth, height: window.innerHeight };
  const card = floating
    ? placeFloating(rect, cardHeight, view)
    : placeCard(rect, cardHeight, view);

  const dim = `rgba(4,5,7,${entered ? 0.74 : 0})`;

  return (
    // Acima da tela de treino (90) e abaixo do Histórico, dos Datasets e da Fila (100).
    <div style={{ position: "fixed", inset: 0, zIndex: 95, pointerEvents: "none" }}>
      {floating ? null : rect ? (
        <>
          <Shade dim={dim} animate={animate} style={{ left: 0, top: 0, right: 0, height: Math.max(0, rect.top - 6) }} />
          <Shade dim={dim} animate={animate} style={{ left: 0, top: rect.bottom + 6, right: 0, bottom: 0 }} />
          <Shade dim={dim} animate={animate} style={{ left: 0, top: rect.top - 6, width: Math.max(0, rect.left - 6), height: rect.height + 12 }} />
          <Shade dim={dim} animate={animate} style={{ left: rect.right + 6, top: rect.top - 6, right: 0, height: rect.height + 12 }} />
          <div
            style={{
              position: "fixed",
              left: rect.left - 6,
              top: rect.top - 6,
              width: rect.width + 12,
              height: rect.height + 12,
              borderRadius: 14,
              border: "1px solid var(--accent-vf)",
              boxShadow: "0 0 26px var(--accent-glow)",
              pointerEvents: "none",
              transition: animate ? MOVE : "none",
            }}
          />
        </>
      ) : (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: dim,
            transition: "background 500ms ease",
            pointerEvents: "none",
          }}
        />
      )}

      <div
        ref={cardRef}
        role="dialog"
        aria-label={t.guidedTour.dialogLabel}
        style={{
          position: "fixed",
          left: card.left,
          top: card.top,
          width: floating ? FLOATING_WIDTH : CARD_WIDTH,
          maxWidth: "calc(100vw - 40px)",
          padding: "20px 22px 18px",
          background: "rgba(10,12,16,0.94)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 16,
          boxShadow: "0 30px 90px rgba(0,0,0,0.62), inset 0 0 26px rgba(255,255,255,0.02)",
          pointerEvents: "auto",
          opacity: entered ? 1 : 0,
          transform: `translateY(${entered ? 0 : 10}px)`,
          transition: `${MOVE}, opacity 380ms ease, transform 460ms cubic-bezier(.16,.84,.24,1)`,
        }}
      >
        <button
          type="button"
          onClick={finish}
          aria-label={t.guidedTour.closeLabel}
          title={t.guidedTour.closeLabel}
          style={{
            position: "absolute",
            top: 12,
            right: 12,
            width: 26,
            height: 26,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "transparent",
            border: "1px solid transparent",
            borderRadius: 8,
            color: "var(--vf-text-muted)",
            fontSize: 14,
            lineHeight: 1,
            cursor: "pointer",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = "var(--vf-panel-stroke)";
            e.currentTarget.style.color = "var(--vf-text)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = "transparent";
            e.currentTarget.style.color = "var(--vf-text-muted)";
          }}
        >
          ✕
        </button>

        {current ? (
          <StepBody step={step} steps={steps} title={current.title} body={current.body} />
        ) : (
          <InviteBody />
        )}

        {current?.action && (
          <div style={{ marginTop: 14 }}>
            <button
              type="button"
              onClick={() => void runAction()}
              disabled={shownAction?.busy === true}
              style={{
                ...secondaryStyle,
                borderColor: "var(--accent-vf)",
                opacity: shownAction?.busy ? 0.55 : 1,
              }}
            >
              {current.action.label}
            </button>
            {shownAction?.result && (
              <div
                role="status"
                style={{
                  ...noteStyle,
                  color: shownAction.result.ok
                    ? "var(--vf-text-dim)"
                    : "oklch(0.85 0.14 22)",
                }}
              >
                {shownAction.result.message}
              </div>
            )}
          </div>
        )}

        {current?.waitFor && (
          <div role="status" style={{ ...noteStyle, display: "flex", gap: 8 }}>
            <span
              style={{
                marginTop: 5,
                width: 6,
                height: 6,
                flexShrink: 0,
                borderRadius: "50%",
                background: open ? "oklch(0.78 0.18 150)" : "var(--accent-vf)",
                animation: open ? "none" : "pulse-dot 2.6s ease-in-out infinite",
              }}
            />
            <span>{open ? t.guidedTour.gateDone : current.waitHint}</span>
          </div>
        )}

        {/* O convite oferece cada guia do registro, lado a lado; o último é o destaque. */}
        {!current && (
          <div style={{ marginTop: 20, display: "flex", gap: 10 }}>
            {GUIDES.map((g, i) => (
              <button
                key={g.id}
                type="button"
                onClick={() => choose(g.id)}
                style={{
                  ...(i === GUIDES.length - 1 ? primaryStyle : secondaryStyle),
                  flex: 1,
                  whiteSpace: "nowrap",
                  padding: "10px 8px",
                }}
              >
                {g.title(t)}
                {i === GUIDES.length - 1 ? " →" : ""}
              </button>
            ))}
          </div>
        )}

        <div
          style={{
            marginTop: current ? 20 : 8,
            display: "flex",
            alignItems: "center",
            gap: 10,
          }}
        >
          <button
            type="button"
            onClick={finish}
            style={ghostStyle}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--vf-text-dim)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--vf-text-muted)")}
          >
            {current ? t.common.skip : t.guidedTour.notNow}
          </button>
          <div style={{ flex: 1 }} />
          {current ? (
            <>
              {step > 0 && (
                <button
                  type="button"
                  onClick={() => setStep((s) => s - 1)}
                  style={secondaryStyle}
                >
                  {t.common.back}
                </button>
              )}
              <button
                type="button"
                onClick={() => (last ? finish() : setStep((s) => s + 1))}
                disabled={!open}
                style={{
                  ...primaryStyle,
                  opacity: open ? 1 : 0.4,
                  cursor: open ? "pointer" : "not-allowed",
                }}
              >
                {last ? t.guidedTour.finish : t.guidedTour.next}
              </button>
            </>
          ) : null}
        </div>
      </div>
    </div>
  );
}



/** Um dos quatro painéis que escurecem tudo o que não é o alvo do passo. */
function Shade({
  dim,
  animate,
  style,
}: {
  dim: string;
  animate: boolean;
  style: React.CSSProperties;
}) {
  return (
    <div
      style={{
        position: "fixed",
        background: dim,
        pointerEvents: "none",
        transition: animate ? `${MOVE}, background 500ms ease` : "background 500ms ease",
        ...style,
      }}
    />
  );
}


function InviteBody() {
  const t = useT();
  return (
    <>
      <div style={eyebrowStyle}>{t.guidedTour.eyebrow}</div>
      <div style={titleStyle}>{t.guidedTour.inviteTitle}</div>
      <p style={bodyStyle}>{t.guidedTour.inviteBody}</p>
    </>
  );
}

function StepBody({
  step,
  steps,
  title,
  body,
}: {
  step: number;
  steps: GuideStep[];
  title: string;
  body: string;
}) {
  return (
    <>
      <div
        style={{
          ...eyebrowStyle,
          display: "flex",
          alignItems: "center",
          gap: 10,
        }}
      >
        <span>
          {String(step + 1).padStart(2, "0")} / {String(steps.length).padStart(2, "0")}
        </span>
        <span style={{ display: "flex", gap: 5, alignItems: "center" }}>
          {steps.map((_, i) => (
            <span
              key={i}
              style={{
                width: i === step ? 16 : 5,
                height: 5,
                borderRadius: 999,
                background: i <= step ? "var(--accent-vf)" : "rgba(255,255,255,0.14)",
                boxShadow: i === step ? "0 0 10px var(--accent-glow)" : "none",
                transition: "width 380ms cubic-bezier(.16,.84,.24,1), background 380ms ease",
              }}
            />
          ))}
        </span>
      </div>
      <div style={titleStyle}>{title}</div>
      <p style={bodyStyle}>{body}</p>
    </>
  );
}

const eyebrowStyle: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.18em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
  marginBottom: 12,
};

const titleStyle: React.CSSProperties = {
  fontFamily: "var(--font-display)",
  fontSize: 21,
  fontWeight: 600,
  letterSpacing: "-0.01em",
  color: "var(--vf-text)",
  paddingRight: 26,
};

const bodyStyle: React.CSSProperties = {
  margin: "10px 0 0",
  fontSize: 13.5,
  lineHeight: 1.65,
  color: "var(--vf-text-dim)",
  // O último passo do primeiro treino é uma lista de uma linha por tarefa.
  whiteSpace: "pre-line",
};

const noteStyle: React.CSSProperties = {
  marginTop: 12,
  fontSize: 12.5,
  lineHeight: 1.55,
  color: "var(--vf-text-muted)",
};

const ghostStyle: React.CSSProperties = {
  padding: "9px 4px",
  background: "transparent",
  border: "none",
  color: "var(--vf-text-muted)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.12em",
  textTransform: "uppercase",
  cursor: "pointer",
  transition: "color 200ms ease",
};

const secondaryStyle: React.CSSProperties = {
  padding: "9px 16px",
  background: "rgba(255,255,255,0.03)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 10,
  color: "var(--vf-text-dim)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.12em",
  textTransform: "uppercase",
  cursor: "pointer",
};

const primaryStyle: React.CSSProperties = {
  padding: "9px 18px",
  background: "var(--accent-soft)",
  border: "1px solid var(--accent-vf)",
  borderRadius: 10,
  color: "var(--vf-text)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.12em",
  textTransform: "uppercase",
  cursor: "pointer",
  boxShadow: "inset 0 0 16px var(--accent-glow)",
};
