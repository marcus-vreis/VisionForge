/** What the "Começando do zero" screen is made of (ADR-207).
 *
 * The screen is for someone who has never trained a model: chapters read top
 * to bottom, each with a figure. Everything that is data lives here, away from
 * the JSX, so the tests can hold it to account: the chapters and their order,
 * the links that leave the app, and the numbers behind the two drawn figures.
 *
 * The figures are illustrations and say so on the screen. Their numbers come
 * from the closed formulas below, not from a run: nothing here pretends to be a
 * measurement.
 */

/** The chapters, in reading order. The id is the anchor and the i18n key. */
export const LEARN_CHAPTERS = [
  "idea",
  "data",
  "loop",
  "curves",
  "result",
  "watch",
  "stack",
  "more",
] as const;

export type LearnChapterId = (typeof LEARN_CHAPTERS)[number];

/** The id of a chapter's heading, which the rail links to. */
export function chapterAnchor(id: LearnChapterId): string {
  return `learn-${id}`;
}

/* ---------------------------------------------------------------------------
 * The split figure: how a folder of images is divided.
 * ------------------------------------------------------------------------- */

export type SplitPart = "train" | "val" | "test";

/** The share of each part, in the order they are drawn: a common division,
 *  which the caption names as such. */
export const SPLIT_SHARES: Record<SplitPart, number> = { train: 0.7, val: 0.15, test: 0.15 };

/** Which part each of `count` images falls in, keeping the shares. The first
 *  images go to training, the next to validation, the rest to test. */
export function splitParts(count: number): SplitPart[] {
  const trainEnd = Math.round(count * SPLIT_SHARES.train);
  const valEnd = trainEnd + Math.round(count * SPLIT_SHARES.val);
  return Array.from({ length: count }, (_, i) =>
    i < trainEnd ? "train" : i < valEnd ? "val" : "test",
  );
}

/* ---------------------------------------------------------------------------
 * The curves figure: loss per epoch at three learning rates.
 * ------------------------------------------------------------------------- */

export type LearningRateChoice = "low" | "good" | "high";

export const LEARNING_RATE_CHOICES: readonly LearningRateChoice[] = ["low", "good", "high"];

/** Epochs the figure draws. */
export const CURVE_EPOCHS = 30;

export interface CurvePoint {
  epoch: number;
  train: number;
  val: number;
}

/** A small repeatable wobble in [-1, 1], so the curves do not look ruled. */
function wobble(epoch: number, salt: number): number {
  return Math.sin(epoch * 1.7 + salt) * 0.6 + Math.sin(epoch * 0.9 + salt * 2.3) * 0.4;
}

/** The illustrative loss curves of a learning rate.
 *
 * - `low`: both losses fall slowly and are still falling at the last epoch;
 * - `good`: both fall fast, then validation turns upwards while training keeps
 *   falling, which is what overfitting looks like;
 * - `high`: the loss jumps around and never settles.
 */
export function lossCurve(choice: LearningRateChoice, epochs = CURVE_EPOCHS): CurvePoint[] {
  return Array.from({ length: epochs }, (_, i) => {
    const epoch = i + 1;
    const x = i / (epochs - 1);
    if (choice === "low") {
      const train = 1.05 - 0.42 * x + wobble(epoch, 1) * 0.008;
      return { epoch, train, val: train + 0.04 + wobble(epoch, 2) * 0.01 };
    }
    if (choice === "high") {
      const train = 0.95 + wobble(epoch, 3) * 0.22 + (i % 3 === 0 ? 0.18 : 0);
      return { epoch, train, val: train + 0.12 + wobble(epoch, 4) * 0.2 };
    }
    const train = 0.06 + 0.99 * Math.exp(-5.2 * x) + wobble(epoch, 5) * 0.006;
    const rise = x > 0.45 ? (x - 0.45) ** 2 * 1.5 : 0;
    const val = 0.19 + 0.86 * Math.exp(-5.2 * x) + rise + wobble(epoch, 6) * 0.012;
    return { epoch, train, val };
  });
}

/** The epoch with the lowest validation loss: the one whose weights are kept. */
export function bestEpoch(points: readonly CurvePoint[]): number {
  return points.reduce((best, p) => (p.val < best.val ? p : best), points[0]).epoch;
}

/* ---------------------------------------------------------------------------
 * The confusion matrix figure: 60 test images of three classes.
 * ------------------------------------------------------------------------- */

/** Rows are the true class, columns the predicted one. Made up, and labelled so. */
export const CONFUSION_EXAMPLE: readonly (readonly number[])[] = [
  [18, 1, 1],
  [2, 16, 2],
  [0, 3, 17],
];

/** Correct predictions over all predictions. */
export function accuracyOf(matrix: readonly (readonly number[])[]): number {
  const total = matrix.flat().reduce((sum, n) => sum + n, 0);
  const correct = matrix.reduce((sum, row, i) => sum + row[i], 0);
  return total === 0 ? 0 : correct / total;
}

/* ---------------------------------------------------------------------------
 * Links that leave the app. Every one opens in a new tab.
 * ------------------------------------------------------------------------- */

/** What runs behind the interface. The id is the i18n key of its one line. */
export const STACK = [
  { id: "pytorch", name: "PyTorch", url: "https://pytorch.org/" },
  { id: "torchvision", name: "torchvision", url: "https://docs.pytorch.org/vision/stable/" },
  { id: "timm", name: "timm", url: "https://huggingface.co/docs/timm" },
  { id: "ultralytics", name: "Ultralytics", url: "https://docs.ultralytics.com/" },
  { id: "sklearn", name: "scikit-learn", url: "https://scikit-learn.org/stable/" },
  { id: "optuna", name: "Optuna", url: "https://optuna.org/" },
  { id: "onnx", name: "ONNX Runtime", url: "https://onnxruntime.ai/" },
  { id: "fastapi", name: "FastAPI", url: "https://fastapi.tiangolo.com/" },
  { id: "react", name: "React", url: "https://react.dev/" },
  { id: "pydantic", name: "Pydantic", url: "https://docs.pydantic.dev/" },
] as const;

export type StackId = (typeof STACK)[number]["id"];

/** Where to study the field. `lang` is the language of the material itself;
 *  the Portuguese ones are listed first when the interface is in Portuguese. */
export const RESOURCES = [
  {
    id: "d2lPt",
    name: "Dive into Deep Learning",
    url: "https://pt.d2l.ai/",
    lang: "pt",
  },
  {
    id: "crashCourse",
    name: "Machine Learning Crash Course",
    url: "https://developers.google.com/machine-learning/crash-course?hl=pt-br",
    lang: "pt",
  },
  {
    id: "threeBlue",
    name: "3Blue1Brown: Neural networks",
    url: "https://www.3blue1brown.com/topics/neural-networks",
    lang: "en",
  },
  { id: "fastai", name: "Practical Deep Learning", url: "https://course.fast.ai/", lang: "en" },
  { id: "cs231n", name: "CS231n", url: "https://cs231n.stanford.edu/", lang: "en" },
  {
    id: "pytorchTutorials",
    name: "PyTorch Tutorials",
    url: "https://docs.pytorch.org/tutorials/",
    lang: "en",
  },
  {
    id: "deepLearningBook",
    name: "Deep Learning (Goodfellow, Bengio, Courville)",
    url: "https://www.deeplearningbook.org/",
    lang: "en",
  },
] as const;

export type ResourceId = (typeof RESOURCES)[number]["id"];

/** The project's own pages, for what comes after this screen. Whole addresses,
 *  on `main`: a test checks that each file exists in this repository. */
export const PROJECT_DOCS = [
  {
    id: "firstTraining",
    url: "https://github.com/marcus-vreis/VisionForge/blob/main/docs/gui/primeiro-treino.md",
  },
  {
    id: "strategies",
    url: "https://github.com/marcus-vreis/VisionForge/blob/main/docs/gui/estrategias.md",
  },
  {
    id: "afterTraining",
    url: "https://github.com/marcus-vreis/VisionForge/blob/main/docs/gui/depois-do-treino.md",
  },
  {
    id: "datasets",
    url: "https://github.com/marcus-vreis/VisionForge/blob/main/docs/gui/datasets.md",
  },
  {
    id: "features",
    url: "https://github.com/marcus-vreis/VisionForge/blob/main/docs/guias/o-que-o-visionforge-faz.md",
  },
  {
    id: "allGuides",
    url: "https://github.com/marcus-vreis/VisionForge/blob/main/docs/gui/README.md",
  },
] as const;

export type ProjectDocId = (typeof PROJECT_DOCS)[number]["id"];
