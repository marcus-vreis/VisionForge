# Model-card characterization

Inputs and complete PT/EN outputs were captured before extraction from
`115c0bcbea59de9cba57cd76bb89c4cb31d2a83f`.
`classification_e2e.json` is the existing offline browser E2E's run.json;
absolute Windows paths were sanitized to `fixture/<basename>` before rendering.
The other inputs exercise the same contract's optional fields, mixed history
columns, unknown status, stopped CV and PatchCore. They are synthetic contract
fixtures, not experimental results. Golden files have no automatic update mode.

`classification_stopped_e2e.json` is a second real offline E2E run.json, from
`frontend/e2e/.work/outputs/models/e2e_stop/20261023_080451_801587/run.json`.
Its PT/EN goldens were captured from the pre-extraction renderer at `c614434`.
The same path sanitization is applied. Both real inputs were checked against
their source run.json files, and every golden was checked against that renderer
before extraction. The completed input comes from
`frontend/e2e/.work/outputs/models/e2e_offline/20261023_080501_978649/run.json`.
