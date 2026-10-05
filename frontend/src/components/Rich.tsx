import { parseRich } from "../lib/rich";

/** A dictionary sentence with inline marks: `code`, **strong**, __emphasis__.
 *
 * The whole sentence lives in the dictionary, so a translation is free to put
 * the marked words wherever its grammar wants them. Marks do not nest, and
 * identifiers go inside backticks (see lib/rich.ts).
 */
export function Rich({ text }: { text: string }) {
  return (
    <>
      {parseRich(text).map((segment, i) => {
        switch (segment.kind) {
          case "code":
            return <code key={i}>{segment.text}</code>;
          case "strong":
            return <strong key={i}>{segment.text}</strong>;
          case "em":
            return <em key={i}>{segment.text}</em>;
          default:
            return segment.text;
        }
      })}
    </>
  );
}
