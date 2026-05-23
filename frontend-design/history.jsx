// History overlay: list of past trainings with a mini loss curve, file actions
// (modelo / gráficos / inferir), and an inference test panel.

const { useState: useStateH, useMemo: useMemoH } = React;

function MiniCurve({ data, color }) {
  const w = 220, h = 56, pad = 4;
  const max = Math.max(...data);
  const min = Math.min(...data);
  const span = (max - min) || 1;
  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (w - pad * 2);
    const y = pad + (1 - (v - min) / span) * (h - pad * 2);
    return [x, y];
  });
  const d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
  const dFill = d + ` L ${pts[pts.length - 1][0]} ${h - pad} L ${pts[0][0]} ${h - pad} Z`;
  const gradId = `g-${Math.random().toString(36).slice(2, 8)}`;
  return (
    <svg width={w} height={h} style={{ display: 'block' }}>
      <defs>
        <linearGradient id={gradId} x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.45" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={dFill} fill={`url(#${gradId})`} />
      <path d={d} fill="none" stroke={color} strokeWidth="1.4" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

function HistoryOverlay({ open, onClose, onResume }) {
  const [selectedId, setSelectedId] = useStateH(window.HISTORY[0].id);
  const selected = useMemoH(
    () => window.HISTORY.find(r => r.id === selectedId),
    [selectedId]
  );

  if (!open) return null;

  const taskOf = (key) => window.TASKS.find(t => t.key === key);

  return (
    <div
      data-task={selected?.task || 'classification'}
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 100,
        background: 'rgba(4,5,7,0.72)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 24,
        animation: 'overlayIn 220ms ease forwards',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 'min(1180px, 100%)',
          height: 'min(720px, 92vh)',
          background: 'rgba(12,14,18,0.92)',
          border: '1px solid var(--panel-stroke)',
          borderRadius: 18,
          boxShadow: '0 50px 120px rgba(0,0,0,0.7), 0 0 0 1px var(--panel-stroke)',
          display: 'grid',
          gridTemplateColumns: '380px 1fr',
          overflow: 'hidden',
          animation: 'sheetIn 260ms cubic-bezier(0.2, 0.9, 0.2, 1) forwards',
        }}
      >
        {/* LEFT: list */}
        <div style={{
          borderRight: '1px solid var(--panel-stroke)',
          display: 'flex', flexDirection: 'column',
          background: 'rgba(8,10,14,0.6)',
        }}>
          <div style={{
            padding: '20px 22px 16px',
            borderBottom: '1px solid var(--panel-stroke)',
          }}>
            <div style={{
              fontSize: 10, letterSpacing: '0.22em',
              color: 'var(--text-muted)', fontFamily: 'var(--mono)', textTransform: 'uppercase',
              marginBottom: 6,
            }}>// training history</div>
            <div style={{ fontSize: 22, fontWeight: 600, letterSpacing: '-0.01em' }}>Treinamentos recentes</div>
            <div style={{ fontSize: 12, color: 'var(--text-dim)', marginTop: 4, fontFamily: 'var(--mono)' }}>
              {window.HISTORY.length} runs · ~/visionforge/runs
            </div>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
            {window.HISTORY.map(r => {
              const t = taskOf(r.task);
              const active = r.id === selectedId;
              return (
                <button
                  key={r.id}
                  onClick={() => setSelectedId(r.id)}
                  data-task={r.task}
                  style={{
                    display: 'block',
                    width: '100%',
                    textAlign: 'left',
                    padding: '12px 14px',
                    background: active ? 'rgba(255,255,255,0.04)' : 'transparent',
                    border: '1px solid',
                    borderColor: active ? 'var(--panel-stroke)' : 'transparent',
                    borderRadius: 12,
                    marginBottom: 6,
                    position: 'relative',
                  }}
                >
                  {active && (
                    <span style={{
                      position: 'absolute', left: 0, top: 14, bottom: 14, width: 2,
                      background: t.accent, borderRadius: 999,
                      boxShadow: `0 0 8px ${t.accent}`,
                    }} />
                  )}
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: 8,
                    fontSize: 10, letterSpacing: '0.16em', textTransform: 'uppercase',
                    fontFamily: 'var(--mono)', color: 'var(--text-muted)',
                  }}>
                    <span style={{
                      width: 6, height: 6, borderRadius: '50%',
                      background: t.accent, boxShadow: `0 0 8px ${t.accent}`,
                    }} />
                    {t.short} · {r.model}
                    {r.status === 'failed' && (
                      <span style={{
                        marginLeft: 'auto', color: 'oklch(0.74 0.18 22)',
                        fontFamily: 'var(--mono)', fontSize: 9.5, letterSpacing: '0.18em',
                      }}>FAILED</span>
                    )}
                  </div>
                  <div style={{
                    fontFamily: 'var(--mono)', fontSize: 13,
                    color: 'var(--text)', marginTop: 4, marginBottom: 8,
                  }}>{r.name}</div>
                  <div style={{
                    display: 'flex', justifyContent: 'space-between',
                    fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-dim)',
                  }}>
                    <span>{r.finishedAt}</span>
                    <span>{r.finalMetric.value}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* RIGHT: detail */}
        <div style={{
          display: 'flex', flexDirection: 'column',
          overflowY: 'auto',
        }}>
          {selected && <HistoryDetail run={selected} task={taskOf(selected.task)} onClose={onClose} onResume={onResume} />}
        </div>
      </div>
    </div>
  );
}

function HistoryDetail({ run, task, onClose, onResume }) {
  const [tab, setTab] = useStateH('overview'); // overview | inference

  return (
    <div style={{ padding: '20px 26px 28px', display: 'flex', flexDirection: 'column', minHeight: '100%' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
        <div style={{ flex: 1 }}>
          <div style={{
            fontSize: 10, letterSpacing: '0.22em', textTransform: 'uppercase',
            color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 8,
            display: 'flex', alignItems: 'center', gap: 10,
          }}>
            <span style={{
              padding: '4px 8px',
              border: `1px solid ${task.accent}`,
              borderRadius: 999,
              color: task.accent,
              boxShadow: `inset 0 0 12px ${task.accent}33`,
            }}>{task.label}</span>
            <span>{run.id}</span>
            <span>·</span>
            <span>{run.finishedAt}</span>
            <span>·</span>
            <span>{run.duration}</span>
            <span>·</span>
            <span>{run.device}</span>
          </div>
          <div style={{ fontSize: 22, fontWeight: 600, fontFamily: 'var(--mono)' }}>{run.name}</div>
        </div>
        <button onClick={onClose} style={closeBtn()}>×</button>
      </div>

      {/* Tabs */}
      <div style={{
        display: 'flex', gap: 4, marginTop: 18,
        borderBottom: '1px solid var(--panel-stroke)',
      }}>
        {[
          { k: 'overview',  l: 'Overview' },
          { k: 'inference', l: 'Testar modelo' },
        ].map(t => (
          <button
            key={t.k}
            onClick={() => setTab(t.k)}
            style={{
              padding: '10px 16px',
              background: 'transparent',
              border: 'none',
              borderBottom: `2px solid ${tab === t.k ? task.accent : 'transparent'}`,
              color: tab === t.k ? 'var(--text)' : 'var(--text-dim)',
              fontFamily: 'var(--mono)', fontSize: 12,
              letterSpacing: '0.06em', textTransform: 'uppercase',
              marginBottom: -1,
            }}
          >{t.l}</button>
        ))}
      </div>

      {tab === 'overview' && <Overview run={run} task={task} onResume={onResume} />}
      {tab === 'inference' && <InferencePanel run={run} task={task} />}
    </div>
  );
}

function Overview({ run, task, onResume }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18, marginTop: 18 }}>
      {/* metric cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
        <MetricCard label="Loss final" value={run.finalLoss.toFixed(4)} accent={task.accent} />
        <MetricCard label={run.finalMetric.label} value={run.finalMetric.value} accent={task.accent} highlight />
        <MetricCard label="Épocas" value={run.epochs} accent={task.accent} />
        <MetricCard label="Duração" value={run.duration} accent={task.accent} />
      </div>

      {/* big curve */}
      <div style={panelCard()}>
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          marginBottom: 14,
        }}>
          <div>
            <div style={{
              fontSize: 10, letterSpacing: '0.22em', textTransform: 'uppercase',
              color: 'var(--text-muted)', fontFamily: 'var(--mono)',
            }}>Training loss curve</div>
            <div style={{ fontSize: 14, marginTop: 4, color: 'var(--text)' }}>{run.epochs} épocas</div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button style={ghostBtn()}>Exportar PNG</button>
            <button style={ghostBtn()}>Exportar CSV</button>
          </div>
        </div>
        <BigCurve data={run.curve} color={task.accent} />
      </div>

      {/* artifacts */}
      <div style={panelCard()}>
        <div style={{
          fontSize: 10, letterSpacing: '0.22em', textTransform: 'uppercase',
          color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 12,
        }}>Artefatos</div>
        <div style={{ display: 'grid', gap: 8 }}>
          <ArtifactRow icon="◆" name="model.pt"      size="184.2 MB" accent={task.accent} primary />
          <ArtifactRow icon="◇" name="config.yaml"   size="2.1 KB"   accent={task.accent} />
          <ArtifactRow icon="◇" name="metrics.json"  size="14.6 KB"  accent={task.accent} />
          <ArtifactRow icon="◇" name="confusion.png" size="92.3 KB"  accent={task.accent} />
          <ArtifactRow icon="◇" name="train.log"     size="1.8 MB"   accent={task.accent} />
        </div>
      </div>

      <div style={{ display: 'flex', gap: 10, marginTop: 4 }}>
        <button style={primaryBtn()} onClick={() => onResume(run)}>Retomar configuração</button>
        <button style={ghostBtn()}>Clonar como novo run</button>
        <button style={ghostBtn()}>Comparar runs</button>
      </div>
    </div>
  );
}

function MetricCard({ label, value, accent, highlight }) {
  return (
    <div style={{
      padding: 14,
      borderRadius: 12,
      border: '1px solid var(--panel-stroke)',
      background: highlight
        ? `linear-gradient(180deg, ${accent}22 0%, rgba(12,14,18,0.5) 100%)`
        : 'rgba(12,14,18,0.55)',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {highlight && (
        <span style={{
          position: 'absolute', top: 10, right: 10,
          width: 6, height: 6, borderRadius: '50%',
          background: accent, boxShadow: `0 0 10px ${accent}`,
        }} />
      )}
      <div style={{
        fontSize: 10, letterSpacing: '0.18em', textTransform: 'uppercase',
        color: 'var(--text-muted)', fontFamily: 'var(--mono)',
      }}>{label}</div>
      <div style={{
        fontSize: 22, marginTop: 6,
        fontFamily: 'var(--mono)', fontWeight: 600,
        color: highlight ? accent : 'var(--text)',
      }}>{value}</div>
    </div>
  );
}

function BigCurve({ data, color }) {
  const w = 760, h = 220, pad = 28;
  const max = Math.max(...data);
  const min = Math.min(...data);
  const span = (max - min) || 1;
  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (w - pad * 2);
    const y = pad + (1 - (v - min) / span) * (h - pad * 2);
    return [x, y];
  });
  const d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
  const dFill = d + ` L ${pts[pts.length - 1][0]} ${h - pad} L ${pts[0][0]} ${h - pad} Z`;
  const gridLines = 4;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" height={h} preserveAspectRatio="none">
      <defs>
        <linearGradient id="bigGrad" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {Array.from({ length: gridLines + 1 }).map((_, i) => {
        const y = pad + (i / gridLines) * (h - pad * 2);
        return <line key={i} x1={pad} x2={w - pad} y1={y} y2={y} stroke="rgba(255,255,255,0.04)" />;
      })}
      <path d={dFill} fill="url(#bigGrad)" />
      <path d={d} fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" />
      {/* min/max labels */}
      <text x={w - pad} y={pad - 8} fill="var(--text-muted)" textAnchor="end"
        style={{ font: '500 10px var(--mono)', letterSpacing: '0.06em' }}>
        max {max.toFixed(3)}
      </text>
      <text x={w - pad} y={h - pad + 16} fill="var(--text-muted)" textAnchor="end"
        style={{ font: '500 10px var(--mono)', letterSpacing: '0.06em' }}>
        min {min.toFixed(3)}
      </text>
    </svg>
  );
}

function ArtifactRow({ icon, name, size, accent, primary }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 14,
      padding: '10px 14px',
      background: primary ? `${accent}11` : 'rgba(255,255,255,0.02)',
      border: '1px solid',
      borderColor: primary ? `${accent}55` : 'var(--panel-stroke)',
      borderRadius: 10,
    }}>
      <span style={{ color: primary ? accent : 'var(--text-muted)', fontFamily: 'var(--mono)' }}>{icon}</span>
      <span style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--text)' }}>{name}</span>
      <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-muted)' }}>{size}</span>
      <span style={{ flex: 1 }} />
      <button style={ghostBtn()}>↗ abrir</button>
      <button style={ghostBtn()}>⤓ baixar</button>
    </div>
  );
}

function InferencePanel({ run, task }) {
  const [files, setFiles] = useStateH([]);
  const [running, setRunning] = useStateH(false);
  const [results, setResults] = useStateH(null);

  const runInference = () => {
    setRunning(true);
    setResults(null);
    setTimeout(() => {
      setRunning(false);
      setResults(makeFakeResults(task, files.length || 4));
    }, 1400);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18, marginTop: 18 }}>
      <div style={panelCard()}>
        <div style={{
          fontSize: 10, letterSpacing: '0.22em', textTransform: 'uppercase',
          color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 12,
        }}>Testar com novos dados</div>

        <div style={{
          border: '1px dashed var(--panel-stroke)',
          borderRadius: 12,
          padding: '32px 20px',
          textAlign: 'center',
          background: 'rgba(255,255,255,0.015)',
          cursor: 'pointer',
        }}
          onClick={() => setFiles(['amostra_001.jpg', 'amostra_002.jpg', 'amostra_003.jpg', 'amostra_004.jpg'])}
        >
          <div style={{ fontSize: 28, color: task.accent, marginBottom: 8 }}>⤓</div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--text)' }}>
            Arraste arquivos ou clique para selecionar
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-muted)', marginTop: 6 }}>
            .jpg .png .csv .parquet · até 1 GB
          </div>
          {files.length > 0 && (
            <div style={{
              marginTop: 14,
              fontFamily: 'var(--mono)', fontSize: 11, color: task.accent,
            }}>
              {files.length} arquivo(s) carregado(s) · pronto para inferência
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: 10, marginTop: 14 }}>
          <button
            style={{ ...primaryBtn(), opacity: files.length === 0 || running ? 0.55 : 1 }}
            disabled={files.length === 0 || running}
            onClick={runInference}
          >
            {running ? 'Executando…' : '▶ Executar inferência'}
          </button>
          <button style={ghostBtn()} onClick={() => { setFiles([]); setResults(null); }}>Limpar</button>
        </div>
      </div>

      {running && (
        <div style={panelCard()}>
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--text-dim)',
            marginBottom: 10, display: 'flex', alignItems: 'center', gap: 10,
          }}>
            <span style={{
              width: 12, height: 12, borderRadius: '50%',
              border: `2px solid ${task.accent}`,
              borderTopColor: 'transparent',
              animation: 'spin 0.8s linear infinite',
            }} />
            Rodando inferência no modelo {run.model}…
          </div>
          <div style={{
            height: 4, borderRadius: 999, overflow: 'hidden',
            background: 'rgba(255,255,255,0.04)',
          }}>
            <div style={{
              width: '60%', height: '100%',
              background: `linear-gradient(90deg, transparent, ${task.accent}, transparent)`,
              backgroundSize: '200% 100%',
              animation: 'shimmer 1.2s linear infinite',
            }} />
          </div>
        </div>
      )}

      {results && (
        <div style={panelCard()}>
          <div style={{
            fontSize: 10, letterSpacing: '0.22em', textTransform: 'uppercase',
            color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginBottom: 12,
          }}>Resultados ({results.length})</div>
          <div style={{ display: 'grid', gap: 8 }}>
            {results.map((r, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 12,
                padding: '10px 14px',
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid var(--panel-stroke)',
                borderRadius: 10,
              }}>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--text-muted)', width: 24 }}>
                  #{String(i + 1).padStart(2, '0')}
                </span>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'var(--text)', flex: 1 }}>
                  {r.file}
                </span>
                <span style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'var(--text-dim)' }}>
                  {r.label}
                </span>
                <span style={{
                  fontFamily: 'var(--mono)', fontSize: 12,
                  color: task.accent, minWidth: 56, textAlign: 'right',
                }}>
                  {r.score}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function makeFakeResults(task, n) {
  const samples = {
    classification: [
      { label: 'rosa_saudável',       score: '0.978' },
      { label: 'mancha_negra',        score: '0.912' },
      { label: 'oídio',               score: '0.846' },
      { label: 'rosa_saudável',       score: '0.991' },
      { label: 'ferrugem',            score: '0.704' },
    ],
    detection: [
      { label: '3 caixas · pcb-defect',   score: 'mAP 0.91' },
      { label: '1 caixa · solder-bridge', score: 'mAP 0.87' },
      { label: '5 caixas · missing-comp', score: 'mAP 0.79' },
      { label: '2 caixas · pcb-defect',   score: 'mAP 0.84' },
    ],
    regression: [
      { label: 'predito · R$ 489.230', score: 'σ 0.08' },
      { label: 'predito · R$ 612.840', score: 'σ 0.11' },
      { label: 'predito · R$ 338.190', score: 'σ 0.06' },
      { label: 'predito · R$ 724.510', score: 'σ 0.14' },
    ],
    segmentation: [
      { label: 'máscara · 4 classes', score: 'Dice 0.93' },
      { label: 'máscara · 4 classes', score: 'Dice 0.89' },
      { label: 'máscara · 4 classes', score: 'Dice 0.95' },
      { label: 'máscara · 4 classes', score: 'Dice 0.91' },
    ],
  }[task.key];
  return Array.from({ length: n }).map((_, i) => ({
    file: `amostra_${String(i + 1).padStart(3, '0')}.${task.key === 'regression' ? 'csv' : 'jpg'}`,
    ...samples[i % samples.length],
  }));
}

function panelCard() {
  return {
    padding: 18,
    borderRadius: 14,
    background: 'rgba(8,10,14,0.5)',
    border: '1px solid var(--panel-stroke)',
  };
}

function primaryBtn() {
  return {
    padding: '10px 18px',
    border: '1px solid var(--accent)',
    background: 'var(--accent-soft)',
    color: 'var(--text)',
    borderRadius: 10,
    fontFamily: 'var(--mono)', fontSize: 12,
    letterSpacing: '0.08em', textTransform: 'uppercase',
    boxShadow: 'inset 0 0 14px var(--accent-glow), 0 0 24px var(--accent-soft)',
  };
}
function ghostBtn() {
  return {
    padding: '8px 12px',
    border: '1px solid var(--panel-stroke)',
    background: 'transparent',
    color: 'var(--text-dim)',
    borderRadius: 8,
    fontFamily: 'var(--mono)', fontSize: 11,
    letterSpacing: '0.06em',
  };
}
function closeBtn() {
  return {
    width: 36, height: 36, borderRadius: '50%',
    border: '1px solid var(--panel-stroke)',
    background: 'rgba(255,255,255,0.03)',
    color: 'var(--text)', fontSize: 22, lineHeight: 1,
  };
}

window.HistoryOverlay = HistoryOverlay;
window.MiniCurve = MiniCurve;
