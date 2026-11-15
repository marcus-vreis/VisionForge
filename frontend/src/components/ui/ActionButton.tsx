import type { ButtonHTMLAttributes, ReactNode } from "react";

/** The run detail's action button: an icon, a short label, one size.
 *
 * Neutral by default, so a row of them reads as a set; the look lives in
 * `.vf-action-btn` (index.css) because hover and focus cannot be inline. The
 * icon is decorative: the label (or `aria-label`) names the action.
 */
export function ActionButton({
  icon,
  children,
  size = "md",
  className,
  type = "button",
  ...rest
}: Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> & {
  icon?: ReactNode;
  children: ReactNode;
  /** `sm` sits inside a row of text (copy a path, open a folder). */
  size?: "md" | "sm";
}) {
  return (
    <button
      type={type}
      className={["vf-action-btn", size === "sm" ? "vf-action-btn--sm" : "", className ?? ""]
        .filter(Boolean)
        .join(" ")}
      {...rest}
    >
      {icon ? (
        <span className="vf-action-btn__icon" aria-hidden="true">
          {icon}
        </span>
      ) : null}
      <span>{children}</span>
    </button>
  );
}
