import { NumberField, SelectField, Segmented, TextField, Toggle } from "./controls";
import { resolveKind } from "./field-renderer";
import type { ValidationError } from "../hooks/useExperiment";
import type { JsonSchema } from "../types/schema";

/**
 * Generic JSON-Schema form (ADR-058, brick 6).
 *
 * Renders a custom task's Config with the same design-system controls the
 * built-in panels use. Deliberately simpler than `ParamPanel`'s renderer: no
 * inline grid axes (a custom task sweeps through the Sweep card, which can
 * target any field by dot-path) and no per-field label dictionary — the
 * researcher's own `title`/`description` from the Pydantic model is the label
 * and the hint, which is what makes "document your field once" pay off.
 */

interface SchemaFormProps {
  schema: JsonSchema;
  value: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  validationErrors: ValidationError[];
  /** Field names hidden because another surface owns them. */
  omit?: string[];
}

/**
 * Human labels for the blocks every custom task inherits from
 * `BaseTaskConfig`. Without these the card headings would be the Pydantic
 * class names the schema carries as `title` (TASKDATACONFIG, SCHEDULERCONFIG),
 * which is noise to a researcher.
 */
const SECTION_LABELS: Record<string, string> = {
  training: "Treinamento",
  data: "Dataset",
  transforms: "Aumentos & normalização",
  preprocessing: "Pré-processamento (filtros)",
  scheduler: "Learning-rate scheduler",
  output: "Saída",
  model: "Modelo",
};

/** Canonical section order (ADR-059), adapted: the researcher's own fields
 *  come first, then the inherited blocks in the same order every built-in
 *  panel uses. Anything unlisted keeps schema order after these. */
const SECTION_ORDER = ["model", "training", "data", "transforms", "preprocessing", "output"];

const sectionLabel: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.22em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
  marginBottom: 12,
};

const card: React.CSSProperties = {
  background: "var(--vf-panel)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 18,
  padding: 26,
  backdropFilter: "blur(14px)",
};

const grid: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
  gap: 14,
};

/** Follow $ref / anyOf until a concrete schema is reached. */
export function resolveSchema(
  schema: JsonSchema,
  defs: Record<string, JsonSchema>,
): JsonSchema {
  if (schema.$ref) {
    const name = schema.$ref.split("/").pop();
    const target = name ? defs[name] : undefined;
    return target ? resolveSchema(target, defs) : schema;
  }
  if (schema.anyOf) {
    const nonNull = schema.anyOf.find((s) => s.type !== "null");
    if (nonNull) return resolveSchema(nonNull, defs);
  }
  return schema;
}

/** Properties of an object schema that will actually render a control. */
export function visibleChildren(
  schema: JsonSchema,
  defs: Record<string, JsonSchema>,
): [string, JsonSchema][] {
  if (schema.type !== "object" || !schema.properties) return [];
  const out: [string, JsonSchema][] = [];
  for (const [name, child] of Object.entries(schema.properties)) {
    const resolved = resolveSchema(child, defs);
    const kind = resolveKind(name, resolved);
    if (kind === "skip") continue;
    // Recurse: a block whose children are all skipped is itself invisible.
    if (kind === "object" && visibleChildren(resolved, defs).length === 0) continue;
    out.push([name, child]);
  }
  return out;
}

/** Friendly heading: the curated label wins over the Pydantic class name. */
function sectionLabel_(name: string, resolved: JsonSchema): string {
  const curated = SECTION_LABELS[name];
  if (curated) return curated;
  // Pydantic titles a nested model with its class name (TaskDataConfig);
  // for a plain field the title is the researcher's own wording, keep it.
  const title = resolved.title;
  if (!title || /Config$/.test(title)) return name.replace(/_/g, " ");
  return title;
}

/** Canonical order first, then whatever else the schema declared. */
export function orderSections(names: string[]): string[] {
  const known = SECTION_ORDER.filter((n) => names.includes(n));
  const rest = names.filter((n) => !SECTION_ORDER.includes(n));
  return [...known, ...rest];
}

function errorFor(
  errors: ValidationError[],
  path: string[],
): string | undefined {
  return errors.find(
    (e) => e.field.length === path.length && e.field.every((f, i) => f === path[i]),
  )?.message;
}

interface FieldProps {
  name: string;
  schema: JsonSchema;
  defs: Record<string, JsonSchema>;
  value: unknown;
  onChange: (v: unknown) => void;
  errors: ValidationError[];
  path: string[];
}

function SchemaField({
  name,
  schema,
  defs,
  value,
  onChange,
  errors,
  path,
}: FieldProps) {
  const resolved = resolveSchema(schema, defs);
  const kind = resolveKind(name, resolved);
  if (kind === "skip") return null;
  // A nested block with nothing renderable (e.g. PreprocessingConfig, whose
  // only field is a step list the schema form cannot edit) must not leave an
  // empty card behind.
  if (kind === "object" && visibleChildren(resolved, defs).length === 0) {
    return null;
  }

  const label = sectionLabel_(name, resolved);
  const hint = resolved.description;
  const message = errorFor(errors, path);

  const wrap = (control: React.ReactNode) => (
    <div>
      {control}
      {message && (
        <div
          style={{
            marginTop: 4,
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "oklch(0.8 0.15 22)",
          }}
        >
          {message}
        </div>
      )}
    </div>
  );

  if (kind === "object") {
    const children = visibleChildren(resolved, defs);
    const obj = (value ?? {}) as Record<string, unknown>;
    return (
      <div style={{ gridColumn: "1 / -1" }}>
        <div style={sectionLabel}>{label}</div>
        <div style={grid}>
          {children.map(([childName, childSchema]) => (
            <SchemaField
              key={childName}
              name={childName}
              schema={childSchema}
              defs={defs}
              value={obj[childName]}
              onChange={(v) => onChange({ ...obj, [childName]: v })}
              errors={errors}
              path={[...path, childName]}
            />
          ))}
        </div>
      </div>
    );
  }

  if (kind === "toggle") {
    return wrap(
      <Toggle
        label={label}
        value={Boolean(value)}
        onChange={onChange}
        hint={hint}
      />,
    );
  }

  if (kind === "segmented" || kind === "select") {
    const options = (resolved.enum ?? []).map((o) => String(o));
    const current = value === undefined || value === null ? options[0] : String(value);
    // Few options read better as a segmented control; many need a dropdown.
    return wrap(
      options.length > 0 && options.length <= 3 ? (
        <Segmented
          label={label}
          value={current}
          onChange={onChange}
          options={options.map((o) => ({ value: o, label: o }))}
          hint={hint}
        />
      ) : (
        <SelectField
          label={label}
          value={current}
          onChange={onChange}
          options={options}
          hint={hint}
        />
      ),
    );
  }

  if (kind === "number") {
    const isInt = resolved.type === "integer";
    return wrap(
      <NumberField
        label={label}
        value={typeof value === "number" ? value : 0}
        onChange={(v) => onChange(isInt ? Math.round(v) : v)}
        min={resolved.minimum ?? resolved.exclusiveMinimum}
        max={resolved.maximum ?? resolved.exclusiveMaximum}
        step={isInt ? 1 : undefined}
        hint={hint}
      />,
    );
  }

  if (kind === "array-number") {
    const list = Array.isArray(value) ? value : [];
    return wrap(
      <TextField
        label={label}
        value={list.join(", ")}
        onChange={(raw) =>
          onChange(
            raw
              .split(",")
              .map((part) => Number(part.trim()))
              .filter((n) => !Number.isNaN(n)),
          )
        }
        hint={hint ?? "valores separados por vírgula"}
        mono
      />,
    );
  }

  return wrap(
    <TextField
      label={label}
      value={value === undefined || value === null ? "" : String(value)}
      onChange={onChange}
      hint={hint}
      mono
    />,
  );
}

/** Render every top-level property of a Config schema into cards. */
export function SchemaForm({
  schema,
  value,
  onChange,
  validationErrors,
  omit = [],
}: SchemaFormProps) {
  const defs = schema.$defs ?? {};
  const hidden = new Set(omit);

  // Nested objects (training, data, …) become their own card; scalars declared
  // directly on the Config — the researcher's own fields — share the first one.
  const scalars: [string, JsonSchema][] = [];
  const objectByName = new Map<string, JsonSchema>();
  for (const [name, child] of visibleChildren(schema, defs)) {
    if (hidden.has(name)) continue;
    const resolved = resolveSchema(child, defs);
    if (resolved.type === "object") objectByName.set(name, child);
    else scalars.push([name, child]);
  }
  const objects: [string, JsonSchema][] = orderSections([
    ...objectByName.keys(),
  ]).map((name) => [name, objectByName.get(name)!]);

  const setField = (name: string, v: unknown) =>
    onChange({ ...value, [name]: v });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      {scalars.length > 0 && (
        <div style={card}>
          <div style={sectionLabel}>Parâmetros da tarefa</div>
          <div style={grid}>
            {scalars.map(([name, child]) => (
              <SchemaField
                key={name}
                name={name}
                schema={child}
                defs={defs}
                value={value[name]}
                onChange={(v) => setField(name, v)}
                errors={validationErrors}
                path={[name]}
              />
            ))}
          </div>
        </div>
      )}
      {objects.map(([name, child]) => (
        <div key={name} style={card}>
          <div style={grid}>
            <SchemaField
              name={name}
              schema={child}
              defs={defs}
              value={value[name]}
              onChange={(v) => setField(name, v)}
              errors={validationErrors}
              path={[name]}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
