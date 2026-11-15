import type { DeviceInfoResponse } from "../api/client";

export interface DeviceSelection {
  kind: "cpu" | "cuda" | "multi_cuda" | "mps";
  gpu_ids: number[] | null;
}

const CPU: DeviceSelection = { kind: "cpu", gpu_ids: null };

/**
 * The selection to switch to when the stored one cannot run here, or null.
 *
 * Mirrors resolve_device on the server (ADR-192): an unavailable CUDA choice
 * goes to the Apple GPU when there is one, and to the CPU otherwise, so the
 * button shows what will actually train, not what was asked for.
 */
export function correctSelection(
  sel: DeviceSelection,
  info: DeviceInfoResponse,
): DeviceSelection | null {
  if (sel.kind === "cpu") return null;
  if (sel.kind === "mps") return info.mps_available ? null : CPU;
  if (!info.cuda_available) {
    return info.mps_available ? { kind: "mps", gpu_ids: null } : CPU;
  }
  if (sel.kind === "multi_cuda" && info.gpus.length < 2) {
    return { kind: "cuda", gpu_ids: null };
  }
  return null;
}

/** Short label for the selector button. */
export function selectionLabel(
  sel: DeviceSelection,
  info: DeviceInfoResponse | null,
): string {
  if (sel.kind === "cpu") return "CPU";
  if (sel.kind === "mps") return "APPLE GPU";
  if (!info || info.gpus.length === 0) return "CUDA";
  if (sel.kind === "multi_cuda") {
    const ids = sel.gpu_ids ?? info.gpus.map((g) => g.index);
    return `MULTI · ${ids.length} GPUs`;
  }
  const id = sel.gpu_ids?.[0] ?? 0;
  return info.gpus.some((x) => x.index === id) ? `GPU ${id}` : "CUDA";
}

/** Whether the selection runs on an accelerator this machine has. */
export function acceleratorActive(
  sel: DeviceSelection,
  info: DeviceInfoResponse | null,
): boolean {
  if (sel.kind === "cpu") return false;
  if (sel.kind === "mps") return !!info?.mps_available;
  return !!info?.cuda_available;
}
