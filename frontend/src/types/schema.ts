export interface JsonSchema {
  type?: string;
  title?: string;
  description?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  enum?: (string | number | boolean)[];
  default?: unknown;
  const?: unknown;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  minLength?: number;
  items?: JsonSchema;
  anyOf?: JsonSchema[];
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
  format?: string;
}
