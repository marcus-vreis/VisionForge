import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";
import { fetchSession } from "./api/client";
import { LanguageProvider } from "./i18n/LanguageProvider";
import { bootSession } from "./lib/session";

// What this server offers is read first: on a shared server the page hides
// what would answer "not available" (ADR-216).
void bootSession(fetchSession).then(() =>
  createRoot(document.getElementById("root")!).render(
    <StrictMode>
      <LanguageProvider>
        <App />
      </LanguageProvider>
    </StrictMode>,
  ),
);
