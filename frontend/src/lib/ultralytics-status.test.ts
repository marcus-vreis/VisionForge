import { describe, expect, it } from "vitest";
import { missingUltralyticsCommand } from "./ultralytics-status";

const COMMAND = 'pip install "visionforge-studio[detection]"';
const absent = { ultralytics_installed: false, detection_install_command: COMMAND };
const present = { ultralytics_installed: true, detection_install_command: COMMAND };

describe("missingUltralyticsCommand", () => {
  it("gives the install line when Ultralytics is chosen and absent", () => {
    expect(missingUltralyticsCommand("ultralytics", absent)).toBe(COMMAND);
  });
  it("says nothing when it is installed, for Torchvision, or before the answer", () => {
    expect(missingUltralyticsCommand("ultralytics", present)).toBeNull();
    expect(missingUltralyticsCommand("torchvision", absent)).toBeNull();
    expect(missingUltralyticsCommand("ultralytics", null)).toBeNull();
  });
});
