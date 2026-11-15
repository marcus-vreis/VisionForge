import { describe, expect, it } from "vitest";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { JsonSchema } from "../types/schema";
import {
  freshSeeds,
  importOutcome,
  planRerun,
  rerunOptions,
  rerunRequest,
  runFamily,
  takePendingImport,
  type PendingImport,
  type RerunSource,
} from "./history-rerun";
import { regressionFormFromPayload } from "./regression-models";
import { buildReplicatesPayload, seedFieldsFor, seedsProblem } from "./replicates-form";
import {
  checkImportedConfig,
  importIntoForm,
  importReviewMessage,
  parseYamlToConfig,
  reviewImportedConfig,
  serializeConfigToYaml,
} from "./yaml-config";

/** The tabs App has with no researcher's task registered. */
const BUILT_INS = ["classification", "detection", "regression", "segmentation", "anomaly"];

/** The part of the regression schema the import checks here. */
const REGRESSION_SCHEMA: JsonSchema = {
  type: "object",
  properties: {
    name: { type: "string" },
    task: { type: "string" },
    data: { $ref: "#/$defs/Data" },
    training: { $ref: "#/$defs/Training" },
  },
  required: ["data"],
  $defs: {
    Data: {
      type: "object",
      properties: {
        base_dir: { type: "string" },
        target_columns: { type: "array", items: { type: "string" } },
      },
      required: ["base_dir"],
    },
    Training: {
      type: "object",
      properties: {
        learning_rate: { type: "number", exclusiveMinimum: 0 },
        epochs: { type: "integer", minimum: 1 },
        seed: { type: "integer", minimum: 0 },
      },
    },
  },
};

/** The part of the classification schema the import checks here. */
const CLASSIFICATION_SCHEMA: JsonSchema = {
  type: "object",
  properties: {
    name: { type: "string" },
    block: { type: "string" },
    data: { $ref: "#/$defs/Data" },
    training: { $ref: "#/$defs/Training" },
  },
  required: ["data"],
  $defs: {
    Data: {
      type: "object",
      properties: { base_dir: { type: "string" } },
      required: ["base_dir"],
    },
    Training: {
      type: "object",
      properties: {
        learning_rate: { type: "number", exclusiveMinimum: 0 },
        epochs: { type: "integer", minimum: 1 },
        seed: { type: "integer", minimum: 0 },
      },
    },
  },
};

const CLASSIFICATION_DEFS = CLASSIFICATION_SCHEMA.$defs ?? {};

/** A regression run as `GET /api/runs/{id}` serves it. */
const regressionRun: RerunSource = {
  run_id: "20261110_reg_s42",
  task: "regression",
  config: {
    name: "reg_s42",
    task: "regression",
    data: { base_dir: "C:/datasets/plants", target_columns: ["height"] },
    // Quoted the way a hand-edited YAML would: the import reads it as the
    // backend does, so the form gets the number.
    training: { learning_rate: "0.0005", epochs: 7, seed: 42 },
  },
};

describe("Carregar no formulário (#200)", () => {
  it("opens the regression tab and fills training.learning_rate from the run", () => {
    const request = rerunRequest(pt, regressionRun, BUILT_INS, "load");
    expect(request).not.toBeNull();
    const plan = planRerun(request!, BUILT_INS, 1);

    expect(plan?.tab).toBe("regression");
    expect(plan?.pending.mode).toBe("load");
    expect(plan?.pending.source).toBe("20261110_reg_s42");

    // What the regression panel runs on the pending config: its YAML import.
    const imported = importIntoForm(
      pt,
      plan!.pending.config,
      REGRESSION_SCHEMA,
      regressionFormFromPayload,
    );
    expect("form" in imported).toBe(true);
    const form = (imported as { form: ReturnType<typeof regressionFormFromPayload> }).form;
    expect(form.training.learning_rate).toBe(0.0005);
    expect(form.training.epochs).toBe(7);
    expect(form.data.base_dir).toBe("C:/datasets/plants");

    // A clean load lands on Revisão as a single run.
    expect(importOutcome(plan!.pending, null, true)).toEqual({
      strategy: "simple",
      step: "review",
      seeds: null,
    });
  });

  it("reads the task from the config when the server sent none", () => {
    expect(runFamily({ config: { task: "segmentation" } })).toBe("segmentation");
    expect(runFamily({ config: { block: "grid_search" } })).toBe("classification");
  });

  it("drops the pending import once its panel applied it, never a newer one", () => {
    const first: PendingImport = { id: 1, config: {}, mode: "load", seeds: [], source: "a" };
    const queue = { regression: first };

    expect(takePendingImport(queue, "regression", 1)).toEqual({});
    const newer = { regression: { ...first, id: 2 } };
    expect(takePendingImport(newer, "regression", 1)).toBe(newer);
  });
});

describe("the warning of a config loaded from History (#200)", () => {
  /** A run.json written before a field changed type, plus a key the form has
   *  no field for. */
  const config = {
    name: "exp",
    block: "classification",
    data: { base_dir: "C:/datasets/leaves" },
    training: { learning_rate: 0.001, epochs: "many", seed: 7 },
    legacy_knob: true,
  };
  const run: RerunSource = { run_id: "20261101_exp", config };

  for (const [lang, t] of [
    ["pt", pt],
    ["en", en],
  ] as const) {
    it(`is the YAML import's warning, word for word (${lang})`, () => {
      const request = rerunRequest(t, run, BUILT_INS, "load");
      const plan = planRerun(request!, BUILT_INS, 1);
      expect(plan?.tab).toBe("classification");

      const fromHistory = importReviewMessage(
        t,
        reviewImportedConfig(t, plan!.pending.config, CLASSIFICATION_SCHEMA, CLASSIFICATION_DEFS),
      );
      // The same run exported to a .yaml and imported with the button.
      const fromYaml = importReviewMessage(
        t,
        reviewImportedConfig(
          t,
          parseYamlToConfig(serializeConfigToYaml(config)),
          CLASSIFICATION_SCHEMA,
          CLASSIFICATION_DEFS,
        ),
      );

      expect(fromHistory).not.toBeNull();
      expect(fromHistory).toBe(fromYaml);
      expect(fromHistory).toContain(t.yamlConfig.expectedInteger);
      expect(fromHistory).toContain(t.paramPanel.importDropped(1));
    });
  }

  it("is the same refusal a standalone panel gives its YAML", () => {
    const bad = { ...regressionRun.config, training: { learning_rate: "fast" } };
    const fromHistory = importIntoForm(pt, bad, REGRESSION_SCHEMA, regressionFormFromPayload);
    const fromYaml = checkImportedConfig(
      pt,
      parseYamlToConfig(serializeConfigToYaml(bad)),
      REGRESSION_SCHEMA,
      REGRESSION_SCHEMA.$defs,
    );

    expect("problem" in fromHistory).toBe(true);
    expect(fromYaml).toEqual(fromHistory);
    // Refused: the form keeps what it had and the message waits on Revisão.
    expect(
      importOutcome({ mode: "replicates", seeds: [43] }, (fromHistory as { problem: string }).problem, true),
    ).toEqual({ strategy: null, step: "review", seeds: null });
  });
});

describe("Repetir com N seeds (#200)", () => {
  it("opens Réplicas with n=3 and none of the run's seed", () => {
    const request = rerunRequest(pt, regressionRun, BUILT_INS, "replicates");
    const plan = planRerun(request!, BUILT_INS, 2);

    expect(plan?.tab).toBe("regression");
    const seeds = plan!.pending.seeds;
    expect(seeds).toHaveLength(3);
    expect(seeds).not.toContain(42);
    expect(new Set(seeds).size).toBe(3);

    const outcome = importOutcome(plan!.pending, null, true);
    expect(outcome.strategy).toBe("replicates");
    expect(outcome.step).toBe("strategy");
    expect(outcome.seeds).toEqual(seeds);

    // The card opens on those seeds and sends exactly them.
    const fields = seedFieldsFor(outcome.seeds!);
    expect(fields.mode).toBe("explicit");
    expect(fields.nReplicates).toBe(3);
    expect(seedsProblem(pt, seeds)).toBeNull();
    const payload = buildReplicatesPayload(fields.mode, fields.nReplicates, fields.rawSeeds, "r2");
    expect(payload.seeds).toEqual(seeds);
  });

  it("avoids every seed of a replicate set", () => {
    const group: RerunSource = {
      run_id: "group_1",
      config: { name: "exp", data: { base_dir: "x" }, training: { seed: 42 } },
      group: { kind: "replicates", seeds: [42, 43, 44, 45, 46] },
    };
    const options = rerunOptions(pt, group, BUILT_INS);

    expect(options.canReplicate).toBe(true);
    expect(options.seeds).toHaveLength(3);
    for (const seed of options.seeds) expect(group.group!.seeds).not.toContain(seed);
  });

  it("takes 42 as the run's seed when its config names none", () => {
    expect(freshSeeds([42])).toEqual([43, 44, 45]);
    const noSeed: RerunSource = { run_id: "r", config: { name: "x", training: {} } };
    expect(rerunOptions(pt, noSeed, BUILT_INS).seeds).not.toContain(42);
  });

  it("is not offered for a sweep, which has no one config to repeat", () => {
    const sweep: RerunSource = {
      run_id: "sweep_1",
      config: { name: "exp", block: "grid_search" },
      group: { kind: "sweep", seeds: [] },
    };
    expect(rerunOptions(pt, sweep, BUILT_INS).canReplicate).toBe(false);
    expect(rerunRequest(pt, sweep, BUILT_INS, "replicates")).toBeNull();
    expect(rerunRequest(pt, sweep, BUILT_INS, "load")).not.toBeNull();
  });

  it("keeps the warning on Revisão when the import had one", () => {
    expect(importOutcome({ mode: "replicates", seeds: [43, 44, 45] }, "aviso", false)).toEqual({
      strategy: "replicates",
      step: "review",
      seeds: [43, 44, 45],
    });
  });
});

describe("a researcher's task removed since the run (#200)", () => {
  const customRun: RerunSource = {
    run_id: "20261020_leaf",
    task: "custom:leaf_counter",
    config: { name: "leaf", training: { seed: 1 } },
  };

  it("turns both actions off and says why, in each language", () => {
    for (const t of [pt, en]) {
      const options = rerunOptions(t, customRun, BUILT_INS);
      expect(options.disabledReason).toBe(t.runDetail.rerun.customRemoved("leaf_counter"));
      expect(options.disabledReason).toContain("leaf_counter");
      expect(rerunRequest(t, customRun, BUILT_INS, "load")).toBeNull();
      expect(rerunRequest(t, customRun, BUILT_INS, "replicates")).toBeNull();
    }
  });

  it("never opens a tab for it, even from a request made before the removal", () => {
    const families = [...BUILT_INS, "custom:leaf_counter"];
    const request = rerunRequest(pt, customRun, families, "load");
    expect(request).not.toBeNull();
    expect(planRerun(request!, families, 1)?.tab).toBe("leaf_counter");

    expect(planRerun(request!, BUILT_INS, 2)).toBeNull();
  });

  it("works while the task is registered", () => {
    const options = rerunOptions(pt, customRun, [...BUILT_INS, "custom:leaf_counter"]);
    expect(options.disabledReason).toBeNull();
  });

  it("says when the run kept no configuration", () => {
    const empty: RerunSource = { run_id: "old", config: {} };
    expect(rerunOptions(pt, empty, BUILT_INS).disabledReason).toBe(pt.runDetail.rerun.noConfig);
  });
});
