import { useEffect, useState } from "react";

import { fetchCustomModels } from "../api/client";
import { useT } from "../i18n/useT";
import { BUILTIN_MODEL, customModelOptions } from "../lib/model-source";
import { ROUTE, useRouteOpen } from "../lib/session";
import { FieldLabel, SelectField } from "./controls";
import { Rich } from "./Rich";

export type CustomModelsLoad = "loading" | "ready" | "failed";

interface CustomModelFieldProps {
  /** The `model.custom_model` the form holds, or `BUILTIN_MODEL`. */
  value: string;
  /** Called with a registered name, or `BUILTIN_MODEL` for the builtin one. */
  onChange: (choice: string) => void;
}

/** Picks `model.custom_model` among the models `user_models/` registers
 *  (ADR-048). The list is read each time the Model step opens, so a file added
 *  while the GUI runs shows up. A shared server closes the route (it would run
 *  code a user brings), and there the field is not drawn. */
export function CustomModelField({ value, onChange }: CustomModelFieldProps) {
  const open = useRouteOpen(ROUTE.customModels);
  const [load, setLoad] = useState<CustomModelsLoad>("loading");
  const [models, setModels] = useState<string[]>([]);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    fetchCustomModels()
      .then((names) => {
        if (!alive) return;
        setModels(names);
        setLoad("ready");
      })
      .catch(() => {
        if (alive) setLoad("failed");
      });
    return () => {
      alive = false;
    };
  }, [open]);

  if (!open) return null;
  return <CustomModelFieldView load={load} models={models} value={value} onChange={onChange} />;
}

/** The field once the list is known. Until then, and when it cannot be read,
 *  nothing is drawn: a YAML can still name the model. With nothing registered
 *  it says how to add one instead of offering an empty list. */
export function CustomModelFieldView({
  load,
  models,
  value,
  onChange,
}: CustomModelFieldProps & { load: CustomModelsLoad; models: readonly string[] }) {
  const t = useT();
  if (load !== "ready") return null;
  const words = t.paramPanel.customModel;
  const label = t.paramPanel.otherModel.label.custom_model;

  if (models.length === 0 && value === BUILTIN_MODEL) {
    return (
      <div data-custom-model="none" style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
        <FieldLabel dot>{label}</FieldLabel>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "var(--vf-text-muted)",
            lineHeight: 1.5,
          }}
        >
          <Rich text={words.empty} />
        </span>
      </div>
    );
  }

  return (
    <div data-custom-model={value === BUILTIN_MODEL ? "builtin" : "custom"} style={{ minWidth: 0 }}>
      <SelectField
        label={label}
        help={words.help}
        value={value}
        onChange={onChange}
        options={customModelOptions(models, value, words)}
      />
    </div>
  );
}
