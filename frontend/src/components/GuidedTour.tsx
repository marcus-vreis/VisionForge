import { useCallback, useEffect, useRef, useState } from "react";

import { CARD_WIDTH, TOUR_STEPS, markTourSeen, placeCard } from "../lib/tour";

/** O guia de primeira execução (ADR-104).
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
  onClose: () => void;
}

export function GuidedTour({ invite = false, onClose }: GuidedTourProps) {
  const [step, setStep] = useState(invite ? -1 : 0);
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

  const current = step >= 0 ? TOUR_STEPS[step] : null;
  const anchor = current?.anchor;
  const last = step === TOUR_STEPS.length - 1;

  const finish = useCallback(() => {
    markTourSeen();
    onClose();
  }, [onClose]);

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
    el?.scrollIntoView({ behavior: "smooth", block: "center" });
    const raf = window.requestAnimationFrame(() => {
      setAnimate(true);
      measure();
    });
    const id = window.setTimeout(measure, 380);
    return () => {
      window.cancelAnimationFrame(raf);
      window.clearTimeout(id);
    };
  }, [anchor, measure]);

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
        setStep((s) => (s + 1 >= TOUR_STEPS.length ? s : s + 1));
      } else if (e.key === "ArrowLeft") {
        setStep((s) => (s > 0 ? s - 1 : s));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [finish]);

  // O cartão vai abaixo do alvo, ou acima quando não sobra espaço; o convite e
  // os passos sem alvo ficam no centro.
  const card = placeCard(rect, cardHeight, {
    width: window.innerWidth,
    height: window.innerHeight,
  });

  const dim = `rgba(4,5,7,${entered ? 0.74 : 0})`;

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 60, pointerEvents: "none" }}>
      {rect ? (
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
        aria-label="Guia do VisionForge"
        style={{
          position: "fixed",
          left: card.left,
          top: card.top,
          width: CARD_WIDTH,
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
          aria-label="Fechar o guia"
          title="Fechar o guia"
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
          <StepBody step={step} title={current.title} body={current.body} />
        ) : (
          <InviteBody />
        )}

        <div
          style={{
            marginTop: 20,
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
            {current ? "Pular" : "Agora não"}
          </button>
          <div style={{ flex: 1 }} />
          {step > 0 && (
            <button
              type="button"
              onClick={() => setStep((s) => s - 1)}
              style={secondaryStyle}
            >
              Voltar
            </button>
          )}
          <button
            type="button"
            onClick={() => (last ? finish() : setStep((s) => s + 1))}
            style={primaryStyle}
          >
            {current ? (last ? "Concluir" : "Continuar →") : "Ver o guia →"}
          </button>
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
  return (
    <>
      <div style={eyebrowStyle}>Primeira vez por aqui</div>
      <div style={titleStyle}>Quer uma volta rápida?</div>
      <p style={bodyStyle}>
        Sete paradas curtas pelos pontos principais: onde escolher a tarefa, como
        apontar o dataset, o que já vem decidido para você e onde os resultados
        ficam guardados. Dá para sair a qualquer momento — e o guia continua
        disponível no cabeçalho depois.
      </p>
    </>
  );
}

function StepBody({
  step,
  title,
  body,
}: {
  step: number;
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
          {String(step + 1).padStart(2, "0")} / {String(TOUR_STEPS.length).padStart(2, "0")}
        </span>
        <span style={{ display: "flex", gap: 5, alignItems: "center" }}>
          {TOUR_STEPS.map((s, i) => (
            <span
              key={s.title}
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
