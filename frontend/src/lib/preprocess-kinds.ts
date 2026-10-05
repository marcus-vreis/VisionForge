/** Human-friendly names of the preprocessing filter kinds. They are the filters'
 *  own technical names, so they read the same in every language; the pipeline
 *  builder and the run detail both show them from here. */
export const PREPROCESS_KIND_LABELS: Record<string, string> = {
  gaussian_blur: "Gaussian blur",
  median_blur: "Median blur",
  unsharp: "Unsharp mask",
  edges: "Edges (Sobel)",
  emboss: "Emboss",
  grayscale: "Grayscale",
  equalize: "Equalize (CLAHE)",
  autocontrast: "Autocontrast",
  wavelet: "Wavelet (Haar)",
};
