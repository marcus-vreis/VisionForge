# VisionForge frontend

React 19 + TypeScript + Vite + Tailwind v4 + shadcn/ui. The built bundle is
committed in `../src/visionforge/gui/static` and served by the Python backend, so
end users never need Node.

## Prerequisites

Node.js `^20.19.0 || >=22.12.0` (the range Vite 8 requires, also in the `engines`
field of `package.json`). Then:

```bash
npm ci
```

## Develop with hot reload

Start the backend without opening a tab, then the Vite dev server:

```bash
visionforge gui --no-browser          # terminal 1, http://127.0.0.1:8000
npm run dev                           # terminal 2, in frontend/
```

Vite forwards `/api` to `http://localhost:8000`. If the backend runs elsewhere
(`visionforge gui --port 9000`), set `VF_API_URL` before `npm run dev`:

```bash
VF_API_URL=http://127.0.0.1:9000 npm run dev                # bash
$env:VF_API_URL = "http://127.0.0.1:9000"; npm run dev      # PowerShell
```

## Checks

```bash
npm run lint            # eslint
npx tsc -b              # types (npm run typecheck)
npx vitest run          # unit tests
npm run e2e             # Playwright against the built bundle (see below)
```

`npm run e2e` starts a real `visionforge` server and tests the **built** SPA, not
the dev server: run `npm run build` first or it exercises a stale bundle. Install
the browser once with `npx playwright install chromium`. `VF_E2E_PORT` and
`VF_E2E_EXECUTABLE` override the port (8765) and the executable.

## Rebuild the SPA

```bash
npm run build
```

Writes to `../src/visionforge/gui/static` (emptied first). Commit the regenerated
bundle together with the source change that caused it.
