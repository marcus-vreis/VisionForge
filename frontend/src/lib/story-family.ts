/** Which drawing the "Como funciona" tab shows for a run (ADR-118).
 *
 * There is one drawing per architecture *family*, not per model: a ResNet-18 and a
 * VGG-19 are the same picture, columns of neurons narrowing towards an output, and
 * the picture only has to say what kind of network is learning. The family is read
 * from what the run was submitted with, once, when it is submitted: the training
 * sheet outlives the form it came from, and a tab switched mid-run must not change
 * the picture of the run in flight.
 *
 * Pure on purpose: no DOM, so the table is covered by plain vitest.
 */

/** The drawings. `autoencoder` and `patchcore` are the two faces of the anomaly
 *  task: one trains by epochs, the other has phases. */
export type StoryFamily =
  | "cnn"
  | "transformer"
  | "encoder_decoder"
  | "detector"
  | "autoencoder"
  | "patchcore";

/** A built-in task, or a researcher's own (`custom`). */
export type StoryTask =
  | "classification"
  | "detection"
  | "regression"
  | "segmentation"
  | "anomaly"
  | "custom";

/** What the network's last column is: class scores, or a single number. Regression
 *  is a CNN (or a transformer) with a one-neuron head; only the two drawings with
 *  a plain head read it. */
export type StoryOutput = "classes" | "value";

/** Which number of an `epoch_end` event is the task's headline metric. `primary`
 *  is a researcher's own task, whose metric the event names only as `val_accuracy`
 *  (the compat field every task fills, ADR-058). */
export type MetricKind = "accuracy" | "map50" | "r2" | "miou" | "auroc" | "primary";

export interface StorySpec {
  task: StoryTask;
  family: StoryFamily;
  output: StoryOutput;
  metric: MetricKind;
}

const METRIC_OF_TASK: Record<StoryTask, MetricKind> = {
  classification: "accuracy",
  detection: "map50",
  regression: "r2",
  segmentation: "miou",
  anomaly: "auroc",
  custom: "primary",
};

/** Names that are attention models. The built-in list has `vit_b_16` and `swin_t`;
 *  `timm` (ADR-100) can name anything, so the rest are the families it ships. A
 *  hybrid such as `mobilevit` or `maxvit` counts: attention is what the drawing
 *  teaches. `convnext` does not: it is a convolutional network. */
const TRANSFORMER_NAME = /vit|deit|swin|beit|cait|xcit|pvt|twins|volo|^eva/i;

/** The family of a model name for a task that takes a backbone choice. */
export function familyForModel(task: StoryTask, name: string | null | undefined): StoryFamily {
  switch (task) {
    case "detection":
      // Ultralytics' YOLOs, RT-DETR and torchvision's detectors all read as
      // backbone -> neck -> heads, which is what the drawing shows.
      return "detector";
    case "segmentation":
      // U-Net, DeepLabV3, FCN and LR-ASPP all encode, then decode to a mask.
      return "encoder_decoder";
    case "anomaly":
      return name === "patchcore" ? "patchcore" : "autoencoder";
    case "custom":
      return "cnn";
    default:
      return name && TRANSFORMER_NAME.test(name) ? "transformer" : "cnn";
  }
}

/** What a run shows when nothing says otherwise: a researcher's own architecture
 *  is unknown, so it gets the generic network. */
export const GENERIC_STORY: StorySpec = {
  task: "custom",
  family: "cnn",
  output: "classes",
  metric: "primary",
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function text(value: unknown): string | null {
  return typeof value === "string" && value.trim() !== "" ? value : null;
}

/** The task key a run was submitted under, as the story knows it. */
function storyTask(task: string): StoryTask {
  if (task.startsWith("custom")) return "custom";
  if (
    task === "detection" ||
    task === "regression" ||
    task === "segmentation" ||
    task === "anomaly"
  ) {
    return task;
  }
  // Classification keeps its problem type (`binary`, `multiclass`) where the
  // others keep their own name (lib/task-family.ts).
  return "classification";
}

/** The drawing for a run, from the task it was submitted for and its config.
 *
 * `config` is the task's config as sent (`model.name`, and for classification
 * `model.timm_model`, `model.custom_model`, `model_comparison.model_names`).
 * A model of the researcher's own (`custom_model`) is unknown to the interface and
 * gets the generic network; a `timm` name is read by `TRANSFORMER_NAME`. A model
 * comparison trains several architectures in one job: when they share a family
 * that is the drawing, otherwise the generic one.
 */
export function storySpecFor(task: string, config: unknown): StorySpec {
  const kind = storyTask(task);
  const cfg = asRecord(config);
  const model = asRecord(cfg?.["model"]);
  const output: StoryOutput = kind === "regression" ? "value" : "classes";
  const metric = METRIC_OF_TASK[kind];

  let family: StoryFamily;
  if (kind === "classification" && text(model?.["custom_model"])) {
    family = "cnn";
  } else if (kind === "classification" && text(model?.["timm_model"])) {
    family = familyForModel(kind, text(model?.["timm_model"]));
  } else {
    // Only a comparison block trains the listed architectures: the list can
    // linger in the form after the researcher moved to another block.
    const compared =
      cfg?.["block"] === "model_comparison"
        ? asRecord(cfg["model_comparison"])?.["model_names"]
        : undefined;
    const names = Array.isArray(compared)
      ? compared.filter((n): n is string => typeof n === "string")
      : [];
    if (kind === "classification" && names.length > 0) {
      const families = new Set(names.map((n) => familyForModel(kind, n)));
      family = families.size === 1 ? [...families][0] : "cnn";
    } else {
      family = familyForModel(kind, text(model?.["name"]));
    }
  }
  return { task: kind, family, output, metric };
}
