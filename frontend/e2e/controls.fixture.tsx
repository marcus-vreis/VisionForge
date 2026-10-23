import { useState } from "react";
import { createRoot } from "react-dom/client";
import { LanguageContext } from "../src/i18n/context";
import { pt } from "../src/i18n/pt";
import { NumberField, Segmented, SelectField, TextField, Toggle } from "../src/components/controls";
import "../src/index.css";

/** Mount shared controls with deterministic values and validation feedback. */
export function ControlsFixture() {
  const [text, setText] = useState("");
  const [number, setNumber] = useState(2);
  const [enabled, setEnabled] = useState(false);
  const [choice, setChoice] = useState("adam");
  return <LanguageContext.Provider value={{ lang: "pt", t: pt, locale: "pt-BR", setLang: () => {} }}>
    <section aria-label="Primeiro card">
      <TextField label="CSV de grupos" value={text} onChange={setText}
        hint="Obrigatório" help="Use um arquivo local." error={text ? undefined : "Informe o CSV."} />
      <NumberField label="Épocas" value={number} onChange={setNumber} min={1} max={10} />
      <Toggle label="AMP" value={enabled} onChange={setEnabled} />
      <SelectField label="Modelo" value={choice} onChange={setChoice} options={["adam", "sgd"]} />
      <Segmented label="Otimizador" value={choice} onChange={setChoice} options={["adam", "sgd"]} />
    </section>
    <section aria-label="Segundo card">
      <TextField label="CSV de grupos" value="second.csv" onChange={() => {}} />
    </section>
  </LanguageContext.Provider>;
}

createRoot(document.getElementById("root")!).render(<ControlsFixture />);
