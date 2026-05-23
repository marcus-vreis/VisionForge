// Reusable input controls themed by --accent.

const { useState, useRef, useEffect } = React;

function FieldLabel({ children, hint, dot }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      marginBottom: 8,
      fontSize: 11,
      letterSpacing: '0.14em',
      textTransform: 'uppercase',
      color: 'var(--text-dim)',
      fontFamily: 'var(--mono)',
    }}>
      {dot && <span style={{
        width: 6, height: 6, borderRadius: '50%',
        background: 'var(--accent)',
        boxShadow: '0 0 8px var(--accent-glow)',
      }} />}
      <span>{children}</span>
      {hint && (
        <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', textTransform: 'none', letterSpacing: 0, fontSize: 11 }}>
          {hint}
        </span>
      )}
    </div>
  );
}

function NumberField({ label, value, onChange, min, max, step = 1, suffix, hint }) {
  const onBlur = (e) => {
    let v = parseFloat(e.target.value);
    if (isNaN(v)) v = value;
    if (typeof min === 'number') v = Math.max(min, v);
    if (typeof max === 'number') v = Math.min(max, v);
    onChange(v);
  };
  return (
    <div>
      <FieldLabel dot hint={hint}>{label}</FieldLabel>
      <div style={fieldShellStyle()}>
        <input
          type="text"
          inputMode="decimal"
          defaultValue={value}
          key={value}
          onBlur={onBlur}
          onKeyDown={(e) => { if (e.key === 'Enter') e.target.blur(); }}
          style={inputBareStyle()}
        />
        {suffix && (
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: 12,
            color: 'var(--text-muted)',
            paddingRight: 14,
          }}>{suffix}</span>
        )}
      </div>
    </div>
  );
}

function TextField({ label, value, onChange, placeholder, hint, mono }) {
  return (
    <div>
      <FieldLabel dot hint={hint}>{label}</FieldLabel>
      <div style={fieldShellStyle()}>
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          style={{ ...inputBareStyle(), fontFamily: mono ? 'var(--mono)' : 'var(--sans)' }}
        />
      </div>
    </div>
  );
}

function SelectField({ label, value, onChange, options, hint }) {
  const [open, setOpen] = useState(false);
  const ref = useRef();
  useEffect(() => {
    const onDoc = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, []);
  const current = options.find(o => (typeof o === 'string' ? o : o.value) === value);
  const currentLabel = current ? (typeof current === 'string' ? current : current.label) : value;
  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <FieldLabel dot hint={hint}>{label}</FieldLabel>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          ...fieldShellStyle(),
          width: '100%',
          padding: 0,
          border: open ? '1px solid var(--accent)' : fieldShellStyle().border,
          boxShadow: open ? '0 0 0 4px var(--accent-soft)' : 'none',
          background: 'rgba(12,14,18,0.65)',
        }}
      >
        <span style={{
          padding: '12px 14px',
          flex: 1,
          textAlign: 'left',
          fontFamily: 'var(--mono)',
          fontSize: 13,
          color: 'var(--text)',
          letterSpacing: '0.01em',
        }}>
          {currentLabel}
        </span>
        <span style={{
          paddingRight: 14,
          color: 'var(--text-muted)',
          transform: open ? 'rotate(180deg)' : 'none',
          transition: 'transform 200ms ease',
        }}>▾</span>
      </button>
      {open && (
        <div style={{
          position: 'absolute',
          top: 'calc(100% + 6px)',
          left: 0,
          right: 0,
          background: 'rgba(14,16,22,0.96)',
          backdropFilter: 'blur(12px)',
          border: '1px solid var(--panel-stroke)',
          borderRadius: 10,
          padding: 6,
          zIndex: 30,
          animation: 'fadeUp 160ms ease',
          maxHeight: 260,
          overflowY: 'auto',
          boxShadow: '0 24px 60px rgba(0,0,0,0.55)',
        }}>
          {options.map(opt => {
            const v = typeof opt === 'string' ? opt : opt.value;
            const l = typeof opt === 'string' ? opt : opt.label;
            const sub = typeof opt === 'string' ? null : opt.sub;
            const active = v === value;
            return (
              <button
                key={v}
                onClick={() => { onChange(v); setOpen(false); }}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'flex-start',
                  gap: 2,
                  width: '100%',
                  padding: '10px 12px',
                  background: active ? 'var(--accent-soft)' : 'transparent',
                  border: 'none',
                  borderRadius: 7,
                  textAlign: 'left',
                  color: 'var(--text)',
                  fontFamily: 'var(--mono)',
                  fontSize: 13,
                  transition: 'background 140ms ease',
                }}
                onMouseEnter={(e) => {
                  if (!active) e.currentTarget.style.background = 'rgba(255,255,255,0.04)';
                }}
                onMouseLeave={(e) => {
                  if (!active) e.currentTarget.style.background = 'transparent';
                }}
              >
                <span>{l}</span>
                {sub && <span style={{ fontSize: 10.5, color: 'var(--text-muted)', letterSpacing: 0 }}>{sub}</span>}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Segmented({ label, value, onChange, options, hint }) {
  return (
    <div>
      <FieldLabel dot hint={hint}>{label}</FieldLabel>
      <div style={{
        display: 'flex',
        background: 'rgba(12,14,18,0.65)',
        border: '1px solid var(--panel-stroke)',
        borderRadius: 10,
        padding: 4,
        gap: 2,
      }}>
        {options.map(opt => {
          const v = typeof opt === 'string' ? opt : opt.value;
          const l = typeof opt === 'string' ? opt : opt.label;
          const active = v === value;
          return (
            <button
              key={v}
              onClick={() => onChange(v)}
              style={{
                flex: 1,
                padding: '8px 10px',
                border: 'none',
                borderRadius: 7,
                background: active ? 'var(--accent-soft)' : 'transparent',
                color: active ? 'var(--text)' : 'var(--text-dim)',
                fontFamily: 'var(--mono)',
                fontSize: 12,
                letterSpacing: '0.04em',
                transition: 'all 160ms ease',
                position: 'relative',
              }}
            >
              {active && <span style={{
                position: 'absolute',
                inset: 0,
                borderRadius: 7,
                border: '1px solid var(--accent)',
                boxShadow: 'inset 0 0 12px var(--accent-glow)',
                pointerEvents: 'none',
              }} />}
              {l}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Toggle({ label, value, onChange, hint }) {
  return (
    <div>
      <FieldLabel dot hint={hint}>{label}</FieldLabel>
      <button
        onClick={() => onChange(!value)}
        style={{
          display: 'flex', alignItems: 'center', gap: 10,
          width: '100%',
          padding: '11px 14px',
          background: 'rgba(12,14,18,0.65)',
          border: '1px solid var(--panel-stroke)',
          borderRadius: 10,
        }}
      >
        <span style={{
          position: 'relative',
          width: 32, height: 18, borderRadius: 999,
          background: value ? 'var(--accent)' : 'rgba(255,255,255,0.12)',
          boxShadow: value ? '0 0 10px var(--accent-glow)' : 'none',
          transition: 'all 200ms ease',
        }}>
          <span style={{
            position: 'absolute',
            top: 2, left: value ? 16 : 2,
            width: 14, height: 14, borderRadius: '50%',
            background: '#fff',
            transition: 'left 200ms ease',
          }} />
        </span>
        <span style={{
          fontFamily: 'var(--mono)', fontSize: 12,
          color: value ? 'var(--text)' : 'var(--text-dim)',
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
        }}>{value ? 'ATIVADO' : 'DESATIVADO'}</span>
      </button>
    </div>
  );
}

function fieldShellStyle() {
  return {
    display: 'flex',
    alignItems: 'center',
    background: 'rgba(12,14,18,0.65)',
    border: '1px solid var(--panel-stroke)',
    borderRadius: 10,
    transition: 'border-color 160ms ease, box-shadow 160ms ease',
  };
}

function inputBareStyle() {
  return {
    flex: 1,
    background: 'transparent',
    border: 'none',
    outline: 'none',
    padding: '12px 14px',
    fontFamily: 'var(--mono)',
    fontSize: 13,
    color: 'var(--text)',
    letterSpacing: '0.01em',
    width: '100%',
  };
}

Object.assign(window, {
  NumberField, TextField, SelectField, Segmented, Toggle, FieldLabel,
});
