import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";

// Where `npm run dev` forwards `/api`. Set VF_API_URL when `visionforge gui` runs on
// another host or port (`visionforge gui --port 9000` -> VF_API_URL=http://127.0.0.1:9000).
const apiUrl = process.env.VF_API_URL ?? "http://localhost:8000";

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
      "/api": apiUrl,
    },
  },
});
