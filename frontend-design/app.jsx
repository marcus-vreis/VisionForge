// Main app for VisionForge landing.

const { useState, useEffect, useRef, useMemo } = React;
const { Waves, Particles, TASKS, HISTORY, HistoryOverlay,
  NumberField, TextField, SelectField, Segmented, Toggle, FieldLabel } = window;

function VisionForgeApp() {
  const [activeKey, setActiveKey] = useState('classification');
  const [device, setDevice] = useState('cuda'); // 'cuda' | 'cpu'
  const [gpuName, setGpuName] = useState('NVIDIA RTX 4090 · 24 GB');
  const [showHistory, setShowHistory] = useState(false);
  const [training, setTraining] = useState(null); // {progress, log[]}

  // store params per task — independent state per tab
  const initialState = useMemo(() => {
    const map = {};
    for (const t of TASKS) map[t.key] = { ...t.defaults };
    return map;
  }, []);
  const [params, setParams] = useState(initialState);

  const activeTask = TASKS.find(t => t.key === activeKey);
  const currentParams = params[activeKey];

  const setParam = (key, value) => {
    setParams(prev => ({
      ...prev,
      [activeKey]: { ...prev[activeKey], [key]: value },
    }));
  };

  const runTraining = () => {
    setTraining({ progress: 0, log: [
      '$ visionforge train --task ' + activeTask.short,
      '> initializing runtime · ' + (device === 'cuda' ? 'CUDA' : 'CPU'),
      '> loading dataset · checking integrity…',
      '> dataset ok · 12.482 amostras · split 80/20',
      `> loading model: ${activeTask.models.find(m => m.value === currentParams.modelo).label}`,
      '> warming up · compiling graph…',
      '$ run started',
    ] });
  };

  // animate fake training — start a loop once when training begins, stop when done
  const trainingActive = !!training && !training.finished;
  useEffect(() => {
    if (!trainingActive) return;
    let p = 0;
    let lastEpoch = -1;
    const totalEp = currentParams.epocas;
    const interval = setInterval(() => {
      p += 0.012;
      const epoch = Math.floor(p * totalEp);
      const loss = Math.max(0.05, 2.3 * Math.exp(-3.2 * p) + (Math.random() - 0.5) * 0.06);
      setTraining(prev => {
        if (!prev) return null;
        if (p >= 1) {
          clearInterval(interval);
          return {
            ...prev,
            progress: 1,
            finished: true,
            log: [...prev.log,
              `> epoch ${totalEp}/${totalEp} · loss ${loss.toFixed(4)}`,
              '> saving model.pt · 184.2 MB',
              '$ training complete',
            ].slice(-26),
          };
        }
        const advancedEpoch = epoch !== lastEpoch;
        lastEpoch = epoch;
        const newLog = advancedEpoch
          ? [...prev.log, `> epoch ${epoch + 1}/${totalEp} · loss ${loss.toFixed(4)} · lr ${currentParams.lr}`].slice(-26)
          : prev.log;
        return { ...prev, progress: p, log: newLog };
      });
    }, 320);
    return () => clearInterval(interval);
  }, [trainingActive]);

  // simulate device probe on boot
  useEffect(() => {
    // could swap to CPU randomly for demo
  }, []);

  const resumeFromHistory = (run) => {
    setActiveKey(run.task);
    setShowHistory(false);
  };

  return (
    <div
      className="stage"
      data-task={activeKey}
      style={{
        minHeight: '100vh',
        position: 'relative',
        overflow: 'hidden',
        background:
          `radial-gradient(1100px 600px at 80% -10%, oklch(0.20 0.04 260 / 0.30), transparent 60%),` +
          `radial-gradient(900px 500px at -10% 110%, var(--accent-soft), transparent 60%)`,
        transition: 'background 600ms ease',
      }}
    >
      <Waves />
      <Particles />

      {/* Top bar */}
      <Header device={device} setDevice={setDevice} gpuName={gpuName} />

      {/* Tabs row */}
      <TabBar tasks={TASKS} activeKey={activeKey} setActiveKey={setActiveKey} />

      {/* Body — parameter grid */}
      <main style={{
        position: 'relative',
        zIndex: 2,
        maxWidth: 1280,
        margin: '0 auto',
        padding: '34px 40px 140px',
      }}>
        <TaskHero task={activeTask} />
        <ParamPanel task={activeTask} params={currentParams} setParam={setParam} />
      </main>

      {/* Bottom action bar */}
      <BottomBar
        onHistory={() => setShowHistory(true)}
        onTrain={runTraining}
        device={device}
        setDevice={setDevice}
        gpuName={gpuName}
      />

      {/* History modal */}
      <HistoryOverlay
        open={showHistory}
        onClose={() => setShowHistory(false)}
        onResume={resumeFromHistory}
      />

      {/* Training overlay */}
      {training && (
        <TrainingOverlay
          training={training}
          task={activeTask}
          params={currentParams}
          device={device}
          onClose={() => setTraining(null)}
        />
      )}
    </div>
  );
}

function Header({ device, gpuName }) {
  const now = new Date();
  const dateStr = now.toLocaleDateString('pt-BR', { day: '2-digit', month: 'short' });
  const timeStr = now.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });
  return (
    <header style={{
      position: 'relative',
      zIndex: 3,
      padding: '22px 40px 0',
      maxWidth: 1280,
      margin: '0 auto',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        <Logo />
        <div>
          <div style={{
            fontSize: 22, fontWeight: 700,
            letterSpacing: '-0.01em',
            lineHeight: 1,
          }}>
            Vision<span style={{ color: 'var(--accent)', textShadow: '0 0 12px var(--accent-glow)' }}>Forge</span>
          </div>
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 11,
            color: 'var(--text-muted)', letterSpacing: '0.16em',
            textTransform: 'uppercase', marginTop: 4,
          }}>local ai studio · v0.4.2</div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 11,
          color: 'var(--text-muted)', letterSpacing: '0.12em',
          textTransform: 'uppercase',
        }}>
          {dateStr} · {timeStr}
        </div>

        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '8px 14px',
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid var(--panel-stroke)',
          borderRadius: 999,
        }}>
          <div style={{
            width: 28, height: 28, borderRadius: '50%',
            background: 'linear-gradient(135deg, var(--accent) 0%, oklch(0.55 0.16 260) 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 600,
          }}>RC</div>
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', letterSpacing: '0.16em', textTransform: 'uppercase', fontFamily: 'var(--mono)' }}>Bem-vindo</div>
            <div style={{ fontSize: 13, fontWeight: 500, lineHeight: 1.1 }}>Rafael</div>
          </div>
        </div>
      </div>
    </header>
  );
}

function Logo() {
  return (
    <div style={{
      width: 40, height: 40,
      borderRadius: 10,
      background: 'radial-gradient(circle at 30% 30%, var(--accent-soft), rgba(8,10,14,0.8))',
      border: '1px solid var(--accent)',
      boxShadow: '0 0 20px var(--accent-glow), inset 0 0 14px var(--accent-soft)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      position: 'relative',
    }}>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="4" stroke="var(--accent)" strokeWidth="1.6" />
        <circle cx="12" cy="12" r="9" stroke="var(--accent)" strokeWidth="1" strokeDasharray="2 3" opacity="0.7" />
        <path d="M12 3 V6 M12 18 V21 M3 12 H6 M18 12 H21" stroke="var(--accent)" strokeWidth="1.2" strokeLinecap="round" />
      </svg>
    </div>
  );
}

function TabBar({ tasks, activeKey, setActiveKey }) {
  return (
    <nav style={{
      position: 'relative',
      zIndex: 3,
      maxWidth: 1280,
      margin: '24px auto 0',
      padding: '0 40px',
      borderBottom: '1px solid var(--panel-stroke)',
    }}>
      <div style={{ display: 'flex', gap: 4, marginBottom: -1 }}>
        {tasks.map(t => {
          const active = t.key === activeKey;
          return (
            <button
              key={t.key}
              onClick={() => setActiveKey(t.key)}
              style={{
                padding: '14px 22px',
                background: active ? `linear-gradient(180deg, ${t.accent}11 0%, transparent 100%)` : 'transparent',
                border: 'none',
                borderBottom: `2px solid ${active ? t.accent : 'transparent'}`,
                color: active ? 'var(--text)' : 'var(--text-dim)',
                fontFamily: 'var(--mono)', fontSize: 12,
                letterSpacing: '0.14em', textTransform: 'uppercase',
                position: 'relative',
                transition: 'all 220ms ease',
                cursor: 'pointer',
              }}
            >
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 10 }}>
                <span style={{
                  width: 6, height: 6, borderRadius: '50%',
                  background: t.accent,
                  boxShadow: active ? `0 0 10px ${t.accent}` : 'none',
                  opacity: active ? 1 : 0.5,
                }} />
                {t.label}
              </span>
              {active && (
                <span style={{
                  position: 'absolute', bottom: -1, left: '15%', right: '15%', height: 2,
                  background: t.accent,
                  boxShadow: `0 0 14px ${t.accent}`,
                  borderRadius: 999,
                }} />
              )}
            </button>
          );
        })}
        <button style={{
          padding: '14px 18px',
          background: 'transparent', border: 'none',
          color: 'var(--text-muted)',
          fontFamily: 'var(--mono)', fontSize: 18,
          letterSpacing: '0.2em',
        }}>···</button>
      </div>
    </nav>
  );
}

function TaskHero({ task }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between',
      marginBottom: 26, animation: 'fadeUp 320ms ease forwards',
    }}
      key={task.key}
    >
      <div>
        <div style={{
          fontFamily: 'var(--mono)', fontSize: 11,
          color: 'var(--text-muted)', letterSpacing: '0.22em',
          textTransform: 'uppercase',
        }}>
          // task / {task.short}
        </div>
        <h1 style={{
          fontSize: 44, fontWeight: 600,
          letterSpacing: '-0.025em',
          margin: '8px 0 6px',
          lineHeight: 1.05,
        }}>
          {task.label.split(' ').map((w, i, arr) => (
            <span key={i}>
              {i === arr.length - 1
                ? <span style={{ background: 'linear-gradient(180deg, var(--text) 0%, var(--text-dim) 100%)', WebkitBackgroundClip: 'text', backgroundClip: 'text', color: 'transparent' }}>{w}</span>
                : w + ' '}
            </span>
          ))}
        </h1>
        <p style={{
          color: 'var(--text-dim)', fontSize: 14,
          maxWidth: 540, lineHeight: 1.5, margin: 0,
        }}>{task.description}.</p>
      </div>

      <div style={{
        display: 'flex', gap: 10, fontFamily: 'var(--mono)', fontSize: 11,
        color: 'var(--text-muted)', letterSpacing: '0.08em',
      }}>
        <DataPill label="Dataset" value="rosas_v3.parquet" />
        <DataPill label="Amostras" value="12.482" />
        <DataPill label="Classes" value={task.key === 'regression' ? 'continuous' : '8'} />
      </div>
    </div>
  );
}

function DataPill({ label, value }) {
  return (
    <div style={{
      padding: '8px 14px',
      borderRadius: 10,
      background: 'rgba(8,10,14,0.55)',
      border: '1px solid var(--panel-stroke)',
    }}>
      <div style={{ fontSize: 9.5, letterSpacing: '0.20em', textTransform: 'uppercase', color: 'var(--text-muted)' }}>{label}</div>
      <div style={{ fontSize: 12, color: 'var(--text)', marginTop: 2 }}>{value}</div>
    </div>
  );
}

function ParamPanel({ task, params, setParam }) {
  return (
    <section
      key={task.key}
      style={{
        position: 'relative',
        padding: 28,
        background: 'rgba(10,12,16,0.55)',
        backdropFilter: 'blur(20px) saturate(140%)',
        WebkitBackdropFilter: 'blur(20px) saturate(140%)',
        border: '1px solid var(--panel-stroke)',
        borderRadius: 20,
        boxShadow: '0 30px 80px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04)',
        overflow: 'hidden',
        animation: 'fadeUp 360ms ease forwards',
      }}
    >
      {/* glow corner */}
      <span style={{
        position: 'absolute', top: -80, right: -80,
        width: 280, height: 280, borderRadius: '50%',
        background: 'var(--accent-soft)',
        filter: 'blur(40px)',
        pointerEvents: 'none',
      }} />

      {/* Model selector — sits at the top, larger than other fields */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1.5fr 1fr 1fr', gap: 18,
        marginBottom: 26, position: 'relative',
      }}>
        <ModelSelector task={task} value={params.modelo} onChange={(v) => setParam('modelo', v)} />

        <NumberField
          label="Épocas"
          value={params.epocas}
          onChange={(v) => setParam('epocas', v)}
          min={1}
          max={5000}
          step={1}
          hint="iterações"
        />
        <NumberField
          label="Learning rate"
          value={params.lr}
          onChange={(v) => setParam('lr', v)}
          min={0.000001}
          max={1}
          step={0.0001}
          hint="lr"
        />
      </div>

      {/* Divider */}
      <div style={{
        height: 1, width: '100%',
        background: 'linear-gradient(90deg, transparent, var(--panel-stroke), transparent)',
        marginBottom: 26,
      }} />

      {/* Remaining params */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 18,
      }}>
        {task.params.filter(p => !['epocas', 'lr'].includes(p.key)).map(p => {
          const v = params[p.key];
          const common = { value: v, onChange: (val) => setParam(p.key, val), label: p.label, hint: p.hint };
          if (p.type === 'number')    return <NumberField key={p.key} {...common} min={p.min} max={p.max} step={p.step} />;
          if (p.type === 'segmented') return <Segmented   key={p.key} {...common} options={p.options} />;
          if (p.type === 'toggle')    return <Toggle      key={p.key} {...common} />;
          if (p.type === 'text')      return <TextField   key={p.key} {...common} />;
          return null;
        })}
      </div>

      {/* Dataset row */}
      <div style={{
        marginTop: 26,
        padding: 16,
        background: 'rgba(255,255,255,0.02)',
        border: '1px dashed var(--panel-stroke)',
        borderRadius: 12,
        display: 'flex', alignItems: 'center', gap: 14,
      }}>
        <span style={{
          width: 36, height: 36, borderRadius: 8,
          background: 'var(--accent-soft)',
          border: '1px solid var(--accent)',
          color: 'var(--accent)',
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: 'var(--mono)', fontSize: 16,
        }}>◇</span>
        <div style={{ flex: 1 }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'var(--text-muted)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>
            Dataset
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 13, marginTop: 2 }}>
            ~/visionforge/data/rosas_v3.parquet · 12.482 amostras · 482 MB
          </div>
        </div>
        <button style={{
          padding: '8px 14px',
          background: 'transparent',
          border: '1px solid var(--panel-stroke)',
          borderRadius: 8,
          color: 'var(--text-dim)',
          fontFamily: 'var(--mono)', fontSize: 11, letterSpacing: '0.06em',
        }}>↗ trocar</button>
      </div>
    </section>
  );
}

function ModelSelector({ task, value, onChange }) {
  return (
    <SelectField
      label="Modelos"
      hint="arquitetura"
      value={value}
      onChange={onChange}
      options={task.models}
    />
  );
}

function BottomBar({ onHistory, onTrain, device, setDevice, gpuName }) {
  return (
    <div style={{
      position: 'fixed',
      bottom: 24,
      left: 24, right: 24,
      maxWidth: 1280,
      margin: '0 auto',
      zIndex: 5,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: 12,
      padding: '12px 14px',
      background: 'rgba(8,10,14,0.78)',
      backdropFilter: 'blur(20px)',
      WebkitBackdropFilter: 'blur(20px)',
      border: '1px solid var(--panel-stroke)',
      borderRadius: 16,
      boxShadow: '0 30px 80px rgba(0,0,0,0.5)',
    }}>
      <button onClick={onHistory} style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '10px 16px',
        background: 'rgba(255,255,255,0.025)',
        border: '1px solid var(--panel-stroke)',
        borderRadius: 10,
        color: 'var(--text-dim)',
        fontFamily: 'var(--mono)', fontSize: 12,
        letterSpacing: '0.08em', textTransform: 'uppercase',
      }}>
        <span style={{ fontSize: 14 }}>⤺</span>
        history
        <span style={{
          marginLeft: 4,
          padding: '2px 7px',
          background: 'var(--accent-soft)',
          color: 'var(--accent)',
          borderRadius: 999,
          fontSize: 10,
          letterSpacing: '0.04em',
        }}>{window.HISTORY.length}</span>
      </button>

      <button onClick={onTrain} style={{
        flex: '0 0 auto',
        position: 'relative',
        padding: '14px 38px',
        background: 'linear-gradient(180deg, var(--accent-soft) 0%, rgba(8,10,14,0.4) 100%)',
        border: '1px solid var(--accent)',
        borderRadius: 12,
        color: 'var(--text)',
        fontFamily: 'var(--mono)', fontSize: 13, fontWeight: 600,
        letterSpacing: '0.18em', textTransform: 'uppercase',
        boxShadow: 'inset 0 0 18px var(--accent-glow), 0 0 30px var(--accent-soft)',
        overflow: 'hidden',
      }}>
        <span style={{
          position: 'absolute', inset: 0,
          background: 'linear-gradient(90deg, transparent, var(--accent-soft), transparent)',
          backgroundSize: '200% 100%',
          animation: 'shimmer 3.6s linear infinite',
          opacity: 0.6,
        }} />
        <span style={{ position: 'relative' }}>▶ Treinar</span>
      </button>

      <DeviceIndicator device={device} setDevice={setDevice} gpuName={gpuName} />
    </div>
  );
}

function DeviceIndicator({ device, setDevice, gpuName }) {
  return (
    <button
      onClick={() => setDevice(device === 'cuda' ? 'cpu' : 'cuda')}
      style={{
        display: 'flex', alignItems: 'center', gap: 12,
        padding: '10px 16px',
        background: device === 'cuda' ? 'oklch(0.78 0.18 150 / 0.10)' : 'rgba(255,255,255,0.025)',
        border: '1px solid',
        borderColor: device === 'cuda' ? 'oklch(0.78 0.18 150 / 0.5)' : 'var(--panel-stroke)',
        borderRadius: 999,
        color: 'var(--text)',
        fontFamily: 'var(--mono)', fontSize: 11,
        letterSpacing: '0.12em', textTransform: 'uppercase',
      }}
      title="Clique para alternar dispositivo (mock)"
    >
      <span style={{
        position: 'relative',
        width: 8, height: 8, borderRadius: '50%',
        background: device === 'cuda' ? 'oklch(0.78 0.18 150)' : 'oklch(0.74 0.10 70)',
        boxShadow: device === 'cuda' ? '0 0 12px oklch(0.78 0.18 150 / 0.8)' : 'none',
        animation: 'pulse-dot 1.6s ease-in-out infinite',
      }} />
      <span style={{ color: 'var(--text-dim)' }}>{device === 'cuda' ? 'usando' : 'usando'}</span>
      <span style={{ color: device === 'cuda' ? 'oklch(0.85 0.16 150)' : 'oklch(0.85 0.10 70)', fontWeight: 600 }}>
        {device === 'cuda' ? 'CUDA' : 'CPU'}
      </span>
      {device === 'cuda' && (
        <span style={{ color: 'var(--text-muted)', fontSize: 10, letterSpacing: '0.08em' }}>
          · {gpuName}
        </span>
      )}
    </button>
  );
}

function TrainingOverlay({ training, task, params, device, onClose }) {
  const pct = Math.round(training.progress * 100);
  const finished = training.finished;
  const logRef = useRef();
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [training.log]);

  return (
    <div
      onClick={(e) => { if (e.target === e.currentTarget && finished) onClose(); }}
      style={{
        position: 'fixed', inset: 0, zIndex: 90,
        background: 'rgba(4,5,7,0.78)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        animation: 'overlayIn 220ms ease forwards',
        padding: 24,
      }}
    >
      <div style={{
        width: 'min(720px, 100%)',
        background: 'rgba(12,14,18,0.95)',
        border: '1px solid var(--panel-stroke)',
        borderRadius: 18,
        padding: 28,
        animation: 'sheetIn 260ms cubic-bezier(0.2, 0.9, 0.2, 1) forwards',
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 14,
          marginBottom: 18,
        }}>
          <span style={{
            width: 16, height: 16, borderRadius: '50%',
            border: `2px solid ${task.accent}`,
            borderTopColor: finished ? task.accent : 'transparent',
            animation: finished ? 'none' : 'spin 0.8s linear infinite',
          }} />
          <div style={{ flex: 1 }}>
            <div style={{
              fontFamily: 'var(--mono)', fontSize: 10, letterSpacing: '0.22em',
              textTransform: 'uppercase', color: 'var(--text-muted)',
            }}>
              {finished ? 'training complete' : `training · ${task.label}`}
            </div>
            <div style={{ fontSize: 18, fontWeight: 600, marginTop: 4, fontFamily: 'var(--mono)' }}>
              {task.short}_{task.models.find(m => m.value === params.modelo).value}_{Date.now().toString().slice(-5)}
            </div>
          </div>
          <div style={{
            fontFamily: 'var(--mono)', fontSize: 28, fontWeight: 600,
            color: task.accent,
          }}>{pct}%</div>
        </div>

        <div style={{
          height: 6, borderRadius: 999, overflow: 'hidden',
          background: 'rgba(255,255,255,0.04)',
          marginBottom: 18,
        }}>
          <div style={{
            width: `${pct}%`, height: '100%',
            background: `linear-gradient(90deg, ${task.accent}, ${task.accent}aa)`,
            boxShadow: `0 0 12px ${task.accent}`,
            transition: 'width 320ms ease',
          }} />
        </div>

        <div ref={logRef} style={{
          height: 220,
          padding: 16,
          background: 'rgba(0,0,0,0.45)',
          border: '1px solid var(--panel-stroke)',
          borderRadius: 12,
          fontFamily: 'var(--mono)', fontSize: 12,
          color: 'var(--text-dim)',
          overflowY: 'auto',
          lineHeight: 1.6,
        }}>
          {training.log.map((line, i) => (
            <div key={i} style={{
              color: line.startsWith('$') ? task.accent : line.startsWith('>') ? 'var(--text)' : 'var(--text-muted)',
            }}>{line}</div>
          ))}
        </div>

        <div style={{
          display: 'flex', gap: 10, marginTop: 18,
          justifyContent: 'flex-end',
        }}>
          {!finished && (
            <button onClick={onClose} style={{
              padding: '10px 18px',
              background: 'transparent',
              border: '1px solid var(--panel-stroke)',
              borderRadius: 10,
              color: 'var(--text-dim)',
              fontFamily: 'var(--mono)', fontSize: 11,
              letterSpacing: '0.08em', textTransform: 'uppercase',
            }}>Cancelar</button>
          )}
          {finished && (
            <>
              <button onClick={onClose} style={{
                padding: '10px 18px',
                background: 'transparent',
                border: '1px solid var(--panel-stroke)',
                borderRadius: 10,
                color: 'var(--text-dim)',
                fontFamily: 'var(--mono)', fontSize: 11,
                letterSpacing: '0.08em', textTransform: 'uppercase',
              }}>Fechar</button>
              <button style={{
                padding: '10px 18px',
                background: 'var(--accent-soft)',
                border: '1px solid var(--accent)',
                borderRadius: 10,
                color: 'var(--text)',
                fontFamily: 'var(--mono)', fontSize: 11,
                letterSpacing: '0.08em', textTransform: 'uppercase',
                boxShadow: 'inset 0 0 14px var(--accent-glow)',
              }}>↗ Ver no histórico</button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<VisionForgeApp />);
