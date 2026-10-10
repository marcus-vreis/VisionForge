import { describe, expect, it } from "vitest";

import { makeDefaultAnomalyForm } from "../anomaly-models";
import { makeDefaultDetectionForm } from "../detection-models";
import { makeDefaultRegressionForm } from "../regression-models";
import { makeDefaultSegmentationForm } from "../segmentation-models";
import { SAMPLE_SETUP } from "./first-training";
import {
  sampleAnomalyForm,
  sampleClassificationForm,
  sampleDetectionForm,
  sampleRegressionForm,
  sampleSegmentationForm,
} from "./sample-forms";

const PATH = "C:/work/datasets/exemplo";

describe("a sample dataset landing in each task's form", () => {
  it("fills classification's schema form and touches nothing else", () => {
    const prev = { name: "x", data: { base_dir: "", image_size: 224 }, training: { epochs: 20 } };

    const next = sampleClassificationForm(prev, PATH);

    expect(next).toEqual({ ...prev, data: { base_dir: PATH, image_size: 224 } });
    expect(prev.data.base_dir).toBe("");
  });

  it("builds the data section when the form has none yet", () => {
    expect(sampleClassificationForm({}, PATH)).toEqual({ data: { base_dir: PATH } });
  });

  it("puts detection back on a folder, with one class and the sample's image size", () => {
    const prev = makeDefaultDetectionForm();
    prev.data.source = "yaml";
    prev.data.data_yaml = "C:/other/data.yaml";
    prev.model.num_classes = 80;

    const next = sampleDetectionForm(prev, PATH, SAMPLE_SETUP.detection);

    expect(next.data.source).toBe("folder");
    expect(next.data.base_dir).toBe(PATH);
    expect(next.data.image_size).toBe(128);
    expect(next.model.num_classes).toBe(1);
    expect(next.data.data_yaml).toBe("C:/other/data.yaml");
    expect(next.training).toBe(prev.training);
  });

  it("sets regression's folder and image size, and leaves the CSV names alone", () => {
    const prev = makeDefaultRegressionForm();

    const next = sampleRegressionForm(prev, PATH, SAMPLE_SETUP.regression);

    expect(next.data.base_dir).toBe(PATH);
    expect(next.data.image_size).toBe(64);
    expect(next.data.train_csv).toBe("train.csv");
    expect(next.data.target_columns).toBe("target");
    expect(next.training).toBe(prev.training);
  });

  it("gives segmentation the three classes of the masks, or the model would reject a mask", () => {
    const prev = makeDefaultSegmentationForm();
    expect(prev.model.num_classes).toBe(2);

    const next = sampleSegmentationForm(prev, PATH, SAMPLE_SETUP.segmentation);

    expect(next.model.num_classes).toBe(3);
    expect(next.data.base_dir).toBe(PATH);
    expect(next.data.image_size).toBe(64);
    expect(next.data.ignore_index).toBe(255);
  });

  it("sets anomaly's folder and image size, and keeps the folder names", () => {
    const next = sampleAnomalyForm(makeDefaultAnomalyForm(), PATH, SAMPLE_SETUP.anomaly);

    expect(next.data.base_dir).toBe(PATH);
    expect(next.data.image_size).toBe(64);
    expect([next.data.train_dir, next.data.test_dir, next.data.normal_dir]).toEqual([
      "train",
      "test",
      "good",
    ]);
  });

  it("changes only the folder when the setup is empty", () => {
    const seg = makeDefaultSegmentationForm();
    const det = makeDefaultDetectionForm();

    expect(sampleSegmentationForm(seg, PATH).model).toBe(seg.model);
    expect(sampleSegmentationForm(seg, PATH).data.image_size).toBe(seg.data.image_size);
    expect(sampleDetectionForm(det, PATH).model).toBe(det.model);
    expect(sampleDetectionForm(det, PATH).data.image_size).toBe(det.data.image_size);
  });
});
