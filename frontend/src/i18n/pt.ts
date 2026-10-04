/**
 * Portuguese — the source dictionary. Its shape *is* the type every other
 * language must match, so a key added here and forgotten in en.ts fails the
 * build rather than showing up blank.
 *
 * Texts that carry values are functions: plurals and interpolation are then
 * ordinary TypeScript, checked like any other call.
 *
 * Organised by where the text appears (one section per component or area),
 * with `common` for the words every screen repeats.
 */
export const pt = {
  common: {
    back: "Voltar",
    next: "Continuar",
    skip: "Pular",
    close: "Fechar",
    cancel: "Cancelar",
    save: "Salvar",
    loading: "Carregando…",
  },
  language: {
    label: "Idioma",
    switchTo: "Mudar para inglês",
  },
};

export type Dict = typeof pt;
