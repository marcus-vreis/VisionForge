# Triagem dos achados do Sonar

Decisões individuais sobre achados do SonarCloud que não viraram correção
([#67](https://github.com/marcus-vreis/VisionForge/issues/67)). Cada entrada
traz o fluxo de dados que sustenta a decisão; o `# NOSONAR` fica só na linha
do achado, com a regra e um motivo curto.

## pythonsecurity:S2083 — `gui/api/colab_jobs.py`, `ColabJobs.run_job()`

Análise de `development` de 2026-11-11. Falso positivo.

- **Fonte apontada:** `_RUNNER.read_text()` em `build_script()`, o texto de
  `colab_remote_runner.py`, que vem instalado com o pacote. O `payload` (config
  validada da requisição) entra no mesmo texto, codificado em base64.
- **Sink apontado:** `script.write_text(text)`, com
  `text = build_script(payload)`. O valor contaminado é o **conteúdo**
  gravado, não o caminho.
- **Caminho:** `script = profile_home(slug) / "jobs" / f"{name}.py"`. O `slug`
  chega por `ProfileDep`, que passa por `resolve_profile()` e
  `is_valid_slug()` (`[a-z0-9][a-z0-9_-]{0,N}`, sem separador nem ponto);
  `name` é `session_name()`, prefixo `vf-` mais `datetime.now()`. Nenhum dos
  dois sai de arquivo, da VM ou do payload.
- **Outros valores remotos no mesmo método:** o `bundle.path` do `job_end` só
  vai como argumento remoto de `colab download` e já é recusado fora de
  `REMOTE_ROOT` ou com `..`; o destino local é fixo
  (`script.with_suffix(".zip")`).
- **Teste que fixa o invariante:**
  `tests/gui/test_colab_jobs.py::test_the_script_path_comes_from_the_slug_and_a_timestamp_only`
  (payload e `bundle.path` com `../`; o script fica em
  `<home>/colab/<slug>/jobs/vf-<timestamp>.py` e nada é baixado).
