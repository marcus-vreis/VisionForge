import { describe, expect, it } from "vitest";
import {
  buildDatasetDownloadPayload,
  makeDefaultDatasetForm,
} from "./dataset-download";

describe("dataset-download payload", () => {
  it("torchvision: parses a positive limit, blank → null", () => {
    const form = makeDefaultDatasetForm();
    form.dataset = "cifar10";
    form.out_dir = "/data/cifar";
    expect(buildDatasetDownloadPayload(form)).toEqual({
      provider: "torchvision",
      dataset: "cifar10",
      out_dir: "/data/cifar",
      limit: null,
    });
    form.limit = "50";
    expect(buildDatasetDownloadPayload(form).limit).toBe(50);
    form.limit = "abc";
    expect(buildDatasetDownloadPayload(form).limit).toBeNull();
  });

  it("roboflow: carries version/api_key/format, defaults format to folder", () => {
    const form = makeDefaultDatasetForm();
    form.provider = "roboflow";
    form.dataset = "ws/proj";
    form.out_dir = "/data/rf";
    form.version = "3";
    form.api_key = "KEY";
    form.dataset_format = "";
    const p = buildDatasetDownloadPayload(form);
    expect(p).toMatchObject({
      provider: "roboflow",
      dataset: "ws/proj",
      version: 3,
      api_key: "KEY",
      dataset_format: "folder",
    });
  });

  it("kaggle: only provider/dataset/out_dir", () => {
    const form = makeDefaultDatasetForm();
    form.provider = "kaggle";
    form.dataset = "owner/slug";
    form.out_dir = "/data/kg";
    form.api_key = "should-not-leak";
    expect(buildDatasetDownloadPayload(form)).toEqual({
      provider: "kaggle",
      dataset: "owner/slug",
      out_dir: "/data/kg",
    });
  });

  it("huggingface: carries optional token (blank → null)", () => {
    const form = makeDefaultDatasetForm();
    form.provider = "huggingface";
    form.dataset = "owner/ds";
    form.out_dir = "/data/hf";
    expect(buildDatasetDownloadPayload(form).token).toBeNull();
    form.token = "hf_xxx";
    expect(buildDatasetDownloadPayload(form).token).toBe("hf_xxx");
  });
});
