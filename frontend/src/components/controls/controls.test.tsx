import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../../i18n/context";
import { pt } from "../../i18n/pt";
import { NumberField } from "./NumberField";
import { Segmented } from "./Segmented";
import { SelectField } from "./SelectField";
import { TextField } from "./TextField";
import { Toggle } from "./Toggle";
import { WorkersField } from "./WorkersField";

function render(node: ReactNode) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={{ lang: "pt", t: pt, locale: "pt-BR", setLang: () => {} }}>
      {node}
    </LanguageContext.Provider>,
  );
}

function attribute(tag: string, name: string) {
  return tag.match(new RegExp(` ${name}="([^"]+)"`))?.[1];
}

describe("shared field accessibility", () => {
  it("associates repeated labels with distinct inputs rather than placeholders", () => {
    const html = render(<>
      <TextField label="CSV de grupos" value="" onChange={() => {}} />
      <TextField label="CSV de grupos" value="" onChange={() => {}} />
      <NumberField label="Épocas" value={2} onChange={() => {}} />
    </>);
    const labels = [...html.matchAll(/<label\b[^>]*>/g)].map(([tag]) => attribute(tag, "for"));
    const inputs = [...html.matchAll(/<input\b[^>]*>/g)].map(([tag]) => attribute(tag, "id"));
    expect(labels).toHaveLength(3);
    expect(inputs.every(Boolean)).toBe(true);
    expect(new Set(inputs).size).toBe(3);
    expect(labels).toEqual(inputs);
  });

  it("keeps help, hint and validation text associated when the tooltip is closed", () => {
    const html = render(<TextField label="Dataset" value="" onChange={() => {}}
      help="Use uma pasta local." hint="Obrigatório" error="Pasta inexistente." />);
    const input = html.match(/<input\b[^>]*>/)?.[0] ?? "";
    expect(attribute(input, "aria-invalid")).toBe("true");
    const ids = attribute(input, "aria-describedby")?.split(" ");
    expect(ids).toHaveLength(3);
    for (const [index, text] of ["Obrigatório", "Use uma pasta local.", "Pasta inexistente."].entries()) {
      expect(html).toMatch(new RegExp(`<[^>]+id="${ids?.[index]}"[^>]*>${text}</`));
    }
    expect(html).not.toContain('role="tooltip"');
  });

  it("omits invalid state and dangling descriptions on ordinary fields", () => {
    const html = render(<NumberField label="Épocas" value={1} onChange={() => {}} />);
    expect(html).not.toContain("aria-describedby");
    expect(html).not.toContain("aria-invalid");
  });

  it.each([
    ["number", <NumberField label="Épocas" value={1} onChange={() => {}} help="Ajuda" error="Inválido" />],
    ["select", <SelectField label="Modelo" value="a" options={["a"]} onChange={() => {}} help="Ajuda" error="Inválido" />],
    ["toggle", <Toggle label="AMP" value={false} onChange={() => {}} help="Ajuda" error="Inválido" />],
    ["segmented", <Segmented label="Modo" value="a" options={["a"]} onChange={() => {}} help="Ajuda" error="Inválido" />],
  ])("associates feedback with the %s control", (_name, node) => {
    const html = render(node);
    const control = html.match(/<(?:input|button|div)\b[^>]*aria-invalid="true"[^>]*>/)?.[0] ?? "";
    const ids = attribute(control, "aria-describedby")?.split(" ");
    expect(ids).toHaveLength(2);
    expect(html).toContain(`id="${ids?.[0]}" class="sr-only">Ajuda</span>`);
    expect(html).toContain(`id="${ids?.[1]}"`);
    expect(html).toContain(">Inválido</p>");
  });

  it("labels dropdown and switch triggers and exposes the switch state", () => {
    const html = render(<>
      <SelectField label="Modelo" value="resnet18" options={["resnet18"]} onChange={() => {}} />
      <Toggle label="AMP" value onChange={() => {}} />
      <Toggle label="Shuffle" value={false} onChange={() => {}} />
    </>);
    const targets = [...html.matchAll(/<label\b[^>]*>/g)].map(([tag]) => attribute(tag, "for"));
    expect(targets).toHaveLength(3);
    for (const id of targets) expect(html).toContain(`<button id="${id}"`);
    expect(html).toContain('role="switch" aria-checked="true"');
    expect(html).toContain('role="switch" aria-checked="false"');
    expect(html).toContain('aria-expanded="false"');
  });

  it("names a segmented group and exposes each selected state", () => {
    const html = render(<Segmented label="Otimizador" value="adam" options={["adam", "sgd"]}
      help="Escolha o algoritmo." onChange={() => {}} />);
    const group = html.match(/<div\b[^>]*role="group"[^>]*>/)?.[0] ?? "";
    expect(attribute(group, "aria-labelledby")).toBeTruthy();
    expect(html).toContain(`id="${attribute(group, "aria-labelledby")}"`);
    expect(html).toContain('aria-pressed="true"');
    expect(html).toContain('aria-pressed="false"');
    expect(attribute(group, "aria-describedby")).toBeTruthy();
  });

  it.each([-1, 2])("labels workers in automatic and manual mode (%s)", (value) => {
    const html = render(<WorkersField value={value} onChange={() => {}} />);
    const target = attribute(html.match(/<label\b[^>]*>/)?.[0] ?? "", "for");
    expect(target).toBeTruthy();
    const tag = html.match(new RegExp(`<${value < 0 ? "button" : "input"}[^>]+id="${target}"[^>]*>`))?.[0] ?? "";
    expect(attribute(tag, "aria-describedby")).toBeTruthy();
    expect(html).toContain(`id="${attribute(tag, "aria-describedby")}"`);
  });
});
