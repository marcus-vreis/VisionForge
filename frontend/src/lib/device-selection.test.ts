import { describe, expect, it } from "vitest";
import type { DeviceInfoResponse } from "../api/client";
import {
  acceleratorActive,
  correctSelection,
  selectionLabel,
} from "./device-selection";

const base: DeviceInfoResponse = {
  cuda_available: false,
  cuda_version: null,
  mps_available: false,
  cpu_name: "Apple M3",
  gpus: [],
};
const mac: DeviceInfoResponse = { ...base, mps_available: true };
const twoGpus: DeviceInfoResponse = {
  ...base,
  cpu_name: "x86",
  cuda_available: true,
  cuda_version: "12.8",
  gpus: [
    { index: 0, name: "RTX 5060 Ti", total_memory_mb: 16000, compute_capability: "12.0" },
    { index: 1, name: "RTX 3090", total_memory_mb: 24000, compute_capability: "8.6" },
  ],
};

describe("correctSelection", () => {
  it("moves the CUDA default to the Apple GPU on a Mac", () => {
    expect(correctSelection({ kind: "cuda", gpu_ids: null }, mac)).toEqual({
      kind: "mps",
      gpu_ids: null,
    });
  });

  it("moves CUDA to the CPU when there is no accelerator", () => {
    expect(correctSelection({ kind: "cuda", gpu_ids: null }, base)).toEqual({
      kind: "cpu",
      gpu_ids: null,
    });
  });

  it("moves MPS to the CPU on a machine without it", () => {
    expect(correctSelection({ kind: "mps", gpu_ids: null }, twoGpus)).toEqual({
      kind: "cpu",
      gpu_ids: null,
    });
  });

  it("drops multi-GPU to one GPU when fewer than two exist", () => {
    const one = { ...twoGpus, gpus: twoGpus.gpus.slice(0, 1) };
    expect(correctSelection({ kind: "multi_cuda", gpu_ids: [0, 1] }, one)).toEqual({
      kind: "cuda",
      gpu_ids: null,
    });
  });

  it("returns null when the selection is already valid", () => {
    expect(correctSelection({ kind: "mps", gpu_ids: null }, mac)).toBeNull();
    expect(correctSelection({ kind: "cuda", gpu_ids: [1] }, twoGpus)).toBeNull();
    expect(correctSelection({ kind: "cpu", gpu_ids: null }, base)).toBeNull();
  });
});

describe("selectionLabel", () => {
  it("names each kind", () => {
    expect(selectionLabel({ kind: "cpu", gpu_ids: null }, base)).toBe("CPU");
    expect(selectionLabel({ kind: "mps", gpu_ids: null }, mac)).toBe("APPLE GPU");
    expect(selectionLabel({ kind: "cuda", gpu_ids: [1] }, twoGpus)).toBe("GPU 1");
    expect(selectionLabel({ kind: "multi_cuda", gpu_ids: null }, twoGpus)).toBe(
      "MULTI · 2 GPUs",
    );
  });

  it("says CUDA before the probe answers or for an unknown GPU", () => {
    expect(selectionLabel({ kind: "cuda", gpu_ids: null }, null)).toBe("CUDA");
    expect(selectionLabel({ kind: "cuda", gpu_ids: [7] }, twoGpus)).toBe("CUDA");
  });
});

describe("acceleratorActive", () => {
  it("lights up only for a device this machine has", () => {
    expect(acceleratorActive({ kind: "mps", gpu_ids: null }, mac)).toBe(true);
    expect(acceleratorActive({ kind: "mps", gpu_ids: null }, twoGpus)).toBe(false);
    expect(acceleratorActive({ kind: "cuda", gpu_ids: null }, twoGpus)).toBe(true);
    expect(acceleratorActive({ kind: "cuda", gpu_ids: null }, mac)).toBe(false);
    expect(acceleratorActive({ kind: "cpu", gpu_ids: null }, twoGpus)).toBe(false);
    expect(acceleratorActive({ kind: "mps", gpu_ids: null }, null)).toBe(false);
  });
});
