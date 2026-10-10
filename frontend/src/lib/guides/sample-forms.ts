/** How a sample dataset lands in each task's form (ADR-117). Pure: no DOM, no React.
 *
 * Every task panel keeps its own form shape and stores the dataset folder in its own
 * place, so the guide's button cannot write one field and have all five panels pick
 * it up. These functions are what the app applies to a form when the button (or the
 * 409 that reuses an existing folder) hands it a path.
 */

import type { AnomalyForm } from "../anomaly-models";
import type { DetectionForm } from "../detection-models";
import type { RegressionForm } from "../regression-models";
import type { SegmentationForm } from "../segmentation-models";
import type { SampleSetup } from "./types";

/** Classification's form is schema-driven (`Record<string, unknown>`); the folder is
 *  `data.base_dir`. Nothing else is touched: the panel reads the splits and the
 *  classes off the folder and sets the model up itself. */
export function sampleClassificationForm(
  prev: Record<string, unknown>,
  path: string,
): Record<string, unknown> {
  return {
    ...prev,
    data: { ...((prev["data"] ?? {}) as Record<string, unknown>), base_dir: path },
  };
}

/** A panel left on a `data.yaml` goes back to a folder: the sample is one. */
export function sampleDetectionForm(
  prev: DetectionForm,
  path: string,
  setup: SampleSetup = {},
): DetectionForm {
  return {
    ...prev,
    model:
      setup.numClasses === undefined
        ? prev.model
        : { ...prev.model, num_classes: setup.numClasses },
    data: {
      ...prev.data,
      source: "folder",
      base_dir: path,
      image_size: setup.imageSize ?? prev.data.image_size,
    },
  };
}

export function sampleRegressionForm(
  prev: RegressionForm,
  path: string,
  setup: SampleSetup = {},
): RegressionForm {
  return {
    ...prev,
    data: {
      ...prev.data,
      base_dir: path,
      image_size: setup.imageSize ?? prev.data.image_size,
    },
  };
}

export function sampleSegmentationForm(
  prev: SegmentationForm,
  path: string,
  setup: SampleSetup = {},
): SegmentationForm {
  return {
    ...prev,
    model:
      setup.numClasses === undefined
        ? prev.model
        : { ...prev.model, num_classes: setup.numClasses },
    data: {
      ...prev.data,
      base_dir: path,
      image_size: setup.imageSize ?? prev.data.image_size,
    },
  };
}

export function sampleAnomalyForm(
  prev: AnomalyForm,
  path: string,
  setup: SampleSetup = {},
): AnomalyForm {
  return {
    ...prev,
    data: {
      ...prev.data,
      base_dir: path,
      image_size: setup.imageSize ?? prev.data.image_size,
    },
  };
}
