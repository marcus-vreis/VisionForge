import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { makeDefaultDetectionForm } from "../lib/detection-models";
import { DetectionPanel } from "./DetectionPanel";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => { } };
function render(form = makeDefaultDetectionForm()) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={lang}>
      <DetectionPanel formData={form} setFormData={() => { }} accent="#abc" validationErrors={[]} />
    </LanguageContext.Provider>,
  );
}
function section(html: string, step: string) {
  const content = html.match(new RegExp(`<section data-step="${step}"[^>]*>([\\s\\S]*?)</section>`));
  expect(content, step).not.toBeNull();
  return content![1];
}
describe("DetectionPanel steps", () => {
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
    expect(section(html, "review")).toContain('value="detection_001"');
    for(const step of ["data", "strategy", "model", "training", "image"]) {
      expect(section(html, step)).not.toContain('value="detection_001"');
      expect(section(html, step)).not.toContain(pt.experimentHeader.nameLabel);
    }
  });

  it("moves classes to data, freeze to model, and pixels to image", () => {
    const html = render();
    expect(section(html, "data")).toContain(pt.detectionPanel.dataset.numClasses);
    expect(section(html, "model")).not.toContain(pt.detectionPanel.dataset.numClasses);
    expect(section(html, "model")).toContain(pt.detectionPanel.mechanics.freeze);
    expect(section(html, "training")).not.toContain(pt.detectionPanel.mechanics.freeze);
    expect(section(html, "image")).toContain(pt.detectionPanel.dataset.imageSize);
    expect(section(html, "data")).not.toContain(pt.detectionPanel.dataset.imageSize);
  });
  it("groups all 22 advanced fields once and hides Ultralytics controls for Torchvision", () => {
    const form = makeDefaultDetectionForm();
    form.training.momentum = 0.8;
    const html = render(form);
    const training = section(html, "training");
    expect(training.split(pt.advancedFields.advanced)).toHaveLength(2);
    expect(training).toContain("(22)");
    for(const label of [pt.detectionPanel.training.optimizer, pt.detectionPanel.schedule.lrf, pt.detectionPanel.schedule.boxGain, pt.detectionPanel.mechanics.labelSmoothing]) expect(training).toContain(label);
    form.model.backend = "torchvision";
    form.model.name = "fasterrcnn_resnet50_fpn";
    const torchvision = render(form);
    expect(section(torchvision, "training")).toContain("(6)");
    expect(section(torchvision, "training")).not.toContain(pt.detectionPanel.schedule.boxGain);
    expect(section(torchvision, "model")).not.toContain(pt.detectionPanel.mechanics.freeze);
    expect(section(torchvision, "image")).not.toContain(pt.detectionPanel.augmentation.toggle);
  });
  it("uses the selected YAML path for dataset readiness", () => {
    const form = makeDefaultDetectionForm();
    form.data.source = "yaml";
    form.data.base_dir = "C:/unused";
    expect(section(render(form), "review")).not.toContain("data-ready");
    form.data.data_yaml = "C:/animals/data.yaml";
    expect(section(render(form), "review")).toContain("data-ready");
  });

});
