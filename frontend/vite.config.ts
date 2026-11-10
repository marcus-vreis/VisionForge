import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    outDir: "../src/visionforge/gui/static",
    emptyOutDir: true,
    rolldownOptions: {
      output: {
        // React and the YAML parser change only on a dependency bump: chunks of
        // their own stay cached across app releases (GUI-12). The panels and
        // overlays are split by their `lazy()` imports.
        codeSplitting: {
          groups: [
            { name: "react", test: /node_modules[\\/](react|react-dom|scheduler)[\\/]/ },
            { name: "js-yaml", test: /node_modules[\\/]js-yaml[\\/]/ },
          ],
        },
      },
    },
  },
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
