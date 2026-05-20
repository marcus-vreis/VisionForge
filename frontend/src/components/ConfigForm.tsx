import { useEffect, useState } from "react";
import { fetchSchema } from "../api/client";
import type { ValidationError } from "../hooks/useExperiment";
import type { JsonSchema } from "../types/schema";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Switch } from "./ui/switch";

interface ConfigFormProps {
  onSubmit: (config: Record<string, unknown>) => void;
  disabled?: boolean;
  validationErrors?: ValidationError[];
}

/** Build default values from a JSON Schema recursively. */
function buildDefaults(
  schema: JsonSchema,
  defs: Record<string, JsonSchema>,
): unknown {
  if (schema.$ref) {
    const refName = schema.$ref.split("/").pop()!;
    return buildDefaults(defs[refName], defs);
  }

  if (schema.default !== undefined) return schema.default;

  if (schema.anyOf) {
    const nonNull = schema.anyOf.find((s) => s.type !== "null");
    if (nonNull) return buildDefaults(nonNull, defs);
    return null;
  }

  if (schema.type === "object" && schema.properties) {
    const obj: Record<string, unknown> = {};
    for (const [key, prop] of Object.entries(schema.properties)) {
      obj[key] = buildDefaults(prop, defs);
    }
    return obj;
  }

  if (schema.enum) return schema.enum[0];
  if (schema.type === "string") return "";
  if (schema.type === "integer" || schema.type === "number") return 0;
  if (schema.type === "boolean") return false;
  if (schema.type === "array") return [];

  return null;
}

/** Resolve a schema ref to its definition. */
function resolveSchema(
  schema: JsonSchema,
  defs: Record<string, JsonSchema>,
): JsonSchema {
  if (schema.$ref) {
    const refName = schema.$ref.split("/").pop()!;
    return defs[refName] ?? schema;
  }
  return schema;
}

/** Find the validation error message for a given field path. */
function fieldError(
  errors: ValidationError[],
  path: string[],
): string | undefined {
  return errors.find(
    (e) =>
      e.field.length === path.length &&
      e.field.every((f, i) => f === path[i]),
  )?.message;
}

/** Section labels for config sub-models. */
const SECTION_LABELS: Record<string, string> = {
  model: "Model",
  training: "Training",
  data: "Dataset",
  output: "Output Paths",
  classification: "Classification",
  transforms: "Transforms",
};

/** Render a single field based on its schema type. */
function SchemaField({
  name,
  schema,
  defs,
  value,
  onChange,
  path,
  errors,
}: {
  name: string;
  schema: JsonSchema;
  defs: Record<string, JsonSchema>;
  value: unknown;
  onChange: (val: unknown) => void;
  path: string[];
  errors: ValidationError[];
}) {
  const resolved = resolveSchema(schema, defs);
  const errorMsg = fieldError(errors, path);

  // Handle anyOf (nullable fields like weights_path: Path | None)
  if (resolved.anyOf) {
    const nonNull = resolved.anyOf.find((s) => s.type !== "null");
    if (nonNull) {
      return (
        <SchemaField
          name={name}
          schema={{ ...nonNull, default: resolved.default, title: resolved.title }}
          defs={defs}
          value={value}
          onChange={onChange}
          path={path}
          errors={errors}
        />
      );
    }
  }

  // Nested object → render as a Card section
  if (resolved.type === "object" && resolved.properties) {
    const objVal = (value ?? {}) as Record<string, unknown>;
    return (
      <Card className="border-border/50">
        <CardHeader className="pb-3 pt-4 px-4">
          <CardTitle className="text-sm font-medium">
            {SECTION_LABELS[name] ?? resolved.title ?? name}
          </CardTitle>
        </CardHeader>
        <CardContent className="px-4 pb-4 grid gap-3">
          {Object.entries(resolved.properties).map(([key, propSchema]) => (
            <SchemaField
              key={key}
              name={key}
              schema={propSchema}
              defs={defs}
              value={objVal[key]}
              onChange={(v) => onChange({ ...objVal, [key]: v })}
              path={[...path, key]}
              errors={errors}
            />
          ))}
        </CardContent>
      </Card>
    );
  }

  // Array of numbers (e.g., normalize_mean/std)
  if (resolved.type === "array" && resolved.items?.type === "number") {
    const arrVal = Array.isArray(value) ? value : [];
    return (
      <div className="grid gap-1.5">
        <Label htmlFor={path.join(".")} className="text-xs text-muted-foreground">
          {resolved.title ?? name}
        </Label>
        <Input
          id={path.join(".")}
          value={arrVal.join(", ")}
          onChange={(e) => {
            const parts = e.target.value
              .split(",")
              .map((s) => s.trim())
              .filter((s) => s !== "")
              .map(Number);
            onChange(parts);
          }}
          placeholder="e.g. 0.485, 0.456, 0.406"
          className="h-8 text-sm"
        />
        {errorMsg && (
          <p className="text-xs text-destructive">{errorMsg}</p>
        )}
      </div>
    );
  }

  // Enum → Select dropdown
  if (resolved.enum) {
    return (
      <div className="grid gap-1.5">
        <Label htmlFor={path.join(".")} className="text-xs text-muted-foreground">
          {resolved.title ?? name}
        </Label>
        <Select
          value={String(value ?? resolved.default ?? resolved.enum[0])}
          onValueChange={onChange}
        >
          <SelectTrigger id={path.join(".")} className="h-8 text-sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {resolved.enum.map((opt) => (
              <SelectItem key={String(opt)} value={String(opt)}>
                {String(opt)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {errorMsg && (
          <p className="text-xs text-destructive">{errorMsg}</p>
        )}
      </div>
    );
  }

  // Boolean → Switch
  if (resolved.type === "boolean") {
    return (
      <div className="flex items-center justify-between">
        <Label htmlFor={path.join(".")} className="text-xs text-muted-foreground">
          {resolved.title ?? name}
        </Label>
        <Switch
          id={path.join(".")}
          checked={Boolean(value)}
          onCheckedChange={onChange}
        />
      </div>
    );
  }

  // Number / Integer → Input[number]
  if (resolved.type === "number" || resolved.type === "integer") {
    return (
      <div className="grid gap-1.5">
        <Label htmlFor={path.join(".")} className="text-xs text-muted-foreground">
          {resolved.title ?? name}
        </Label>
        <Input
          id={path.join(".")}
          type="number"
          value={(value as string | number) ?? ""}
          onChange={(e) => {
            const v = e.target.value;
            if (v === "") {
              onChange(undefined);
              return;
            }
            onChange(
              resolved.type === "integer" ? parseInt(v, 10) : parseFloat(v),
            );
          }}
          min={resolved.minimum ?? resolved.exclusiveMinimum}
          step={resolved.type === "integer" ? 1 : "any"}
          className="h-8 text-sm"
        />
        {errorMsg && (
          <p className="text-xs text-destructive">{errorMsg}</p>
        )}
      </div>
    );
  }

  // String → Input[text]
  return (
    <div className="grid gap-1.5">
      <Label htmlFor={path.join(".")} className="text-xs text-muted-foreground">
        {resolved.title ?? name}
      </Label>
      <Input
        id={path.join(".")}
        type="text"
        value={String(value ?? "")}
        onChange={(e) => onChange(e.target.value)}
        className="h-8 text-sm"
      />
      {errorMsg && (
        <p className="text-xs text-destructive">{errorMsg}</p>
      )}
    </div>
  );
}

export function ConfigForm({
  onSubmit,
  disabled,
  validationErrors = [],
}: ConfigFormProps) {
  const [schema, setSchema] = useState<JsonSchema | null>(null);
  const [formData, setFormData] = useState<Record<string, unknown>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSchema()
      .then((s) => {
        setSchema(s);
        const defaults = buildDefaults(s, s.$defs ?? {}) as Record<
          string,
          unknown
        >;
        setFormData(defaults);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8 text-muted-foreground">
        Loading schema...
      </div>
    );
  }

  if (!schema) {
    return (
      <div className="p-8 text-destructive">
        Failed to load configuration schema.
      </div>
    );
  }

  const defs = schema.$defs ?? {};

  // Top-level fields ordering
  const topLevelOrder = [
    "name",
    "task",
    "model",
    "training",
    "data",
    "output",
    "classification",
  ];

  const orderedProps = topLevelOrder.filter(
    (k) => schema.properties?.[k] !== undefined,
  );

  return (
    <form
      className="grid gap-4"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(formData);
      }}
    >
      {orderedProps.map((key) => (
        <SchemaField
          key={key}
          name={key}
          schema={schema.properties![key]}
          defs={defs}
          value={formData[key]}
          onChange={(v) => setFormData((prev) => ({ ...prev, [key]: v }))}
          path={[key]}
          errors={validationErrors}
        />
      ))}

      <Button type="submit" disabled={disabled} className="w-full mt-2">
        {disabled ? "Running..." : "Run Experiment"}
      </Button>
    </form>
  );
}
