import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { makeDefaultSegmentationForm } from "../lib/segmentation-models";
import { SegmentationPanel } from "./SegmentationPanel";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => { } };
function render(form = makeDefaultSegmentationForm()) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={lang}>
      <SegmentationPanel formData={form} setFormData={() => { }} accent="#abc" validationErrors={[]} />
    </LanguageContext.Provider>,
  );
}
function section(html: string, step: string) {
  const content = html.match(new RegExp(`<section data-step="${step}"[^>]*>([\\s\\S]*?)</section>`));
  expect(content, step).not.toBeNull();
  return content![1];
}
describe("SegmentationPanel steps", () => {
  it("keeps all six steps mounted and places the tour targets in their steps", () => {
    const html = render();
    for(const step of ["data", "strategy", "model", "training", "image", "review"]) section(html, step);
    expect(section(html, "data")).toContain('data-tour="dataset"');
    expect(section(html, "data")).toContain('data-tour="dataset-stats"');
    expect(section(html, "training")).toContain('data-tour="training"');
    expect(section(html, "review")).not.toContain('data-tour="training"');
  });
  it("places the experiment name only in review", () => {
    const html = render();
    expect(section(html, "review")).toContain('value="segmentation_001"');
    for(const step of ["data", "strategy", "model", "training", "image"]) {
      expect(section(html, step)).not.toContain('value="segmentation_001"');
      expect(section(html, step)).not.toContain(pt.experimentHeader.nameLabel);
    }
  });

  it("places classes and the ignore-index collision together in data", () => {
    const form = makeDefaultSegmentationForm();
    form.data.ignore_index = 0;
    const html = render(form);
    const data = section(html, "data");
    expect(data).toContain(pt.segmentationPanel.model.numClasses);
    expect(data).toContain(pt.segmentationPanel.dataset.ignoreIndex);
    const warning = pt.segmentationPanel.ignoreIndexCollision(0, form.model.num_classes - 1);
    expect(data).toContain(warning);
    expect(data.indexOf(warning)).toBeGreaterThan(data.indexOf(pt.segmentationPanel.dataset.ignoreIndex));
    expect(section(html, "model")).not.toContain(pt.segmentationPanel.model.numClasses);
    expect(section(html, "review")).not.toContain(warning);
  });
  it("keeps transfer learning in model and image size in image", () => {
    const form = makeDefaultSegmentationForm();
    form.transfer = "fine_tuning";
    const html = render(form);
    expect(section(html, "model")).toContain(pt.taskPanel.transfer.mode);
    expect(section(html, "model")).toContain(pt.taskPanel.transfer.backboneLr);
    expect(section(html, "training")).not.toContain(pt.taskPanel.transfer.mode);
    expect(section(html, "data")).not.toContain(pt.taskPanel.dataset.imageSize);
    expect(section(html, "image")).toContain(pt.taskPanel.dataset.imageSize);
  });

});
