/** Pure form-state → payload helpers for the dataset-download card (ADR-055).
 *  Kept out of the component so the per-provider payload shaping is unit-testable. */

import type { DatasetDownloadRequest } from "../api/client";

export type DatasetProvider =
  | "torchvision"
  | "roboflow"
  | "kaggle"
  | "huggingface";

export const TORCHVISION_DATASETS: { value: string; label: string }[] = [
  { value: "cifar10", label: "CIFAR-10" },
  { value: "cifar100", label: "CIFAR-100" },
  { value: "mnist", label: "MNIST" },
  { value: "fashion_mnist", label: "Fashion-MNIST" },
  { value: "kmnist", label: "KMNIST" },
];

/** Controlled form state for the dataset-download card (all numerics are UI
 *  strings so the fields can be blank). */
export interface DatasetDownloadForm {
  provider: DatasetProvider;
  dataset: string;
  out_dir: string;
  limit: string; // torchvision: images per class (blank = all)
  version: string; // roboflow
  api_key: string; // roboflow
  dataset_format: string; // roboflow
  token: string; // huggingface
}

export function makeDefaultDatasetForm(): DatasetDownloadForm {
  return {
    provider: "torchvision",
    dataset: "cifar10",
    out_dir: "",
    limit: "",
    version: "",
    api_key: "",
    dataset_format: "folder",
    token: "",
  };
}

function _posInt(raw: string): number | null {
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : null;
}

/** Project the form into the wire payload, including only fields the provider
 *  uses (so an untouched roboflow api_key doesn't leak into a torchvision call). */
export function buildDatasetDownloadPayload(
  form: DatasetDownloadForm,
): DatasetDownloadRequest {
  const base: DatasetDownloadRequest = {
    provider: form.provider,
    dataset: form.dataset.trim(),
    out_dir: form.out_dir.trim(),
  };
  if (form.provider === "torchvision") {
    return { ...base, limit: _posInt(form.limit) };
  }
  if (form.provider === "roboflow") {
    return {
      ...base,
      version: _posInt(form.version),
      api_key: form.api_key.trim() || null,
      dataset_format: form.dataset_format.trim() || "folder",
    };
  }
  if (form.provider === "huggingface") {
    return { ...base, token: form.token.trim() || null };
  }
  return base; // kaggle: just provider/dataset/out_dir
}
