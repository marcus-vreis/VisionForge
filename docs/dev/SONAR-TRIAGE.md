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

## python:S7503 — `gui/api/run_events.py`, `RunEventLog.put()`

Análise de `development` de 2026-11-11. Mantido `async`, por contrato.

- **Chamadas:** as threads de treino entregam eventos com
  `asyncio.run_coroutine_threadsafe(queue.put(event), loop)` em
  `execution.py`, `training_routes.py`, `strategies.py`,
  `custom_task_routes.py` e `colab_run_routes.py` (este espera o
  `Future` com timeout); os executores fecham o log com
  `await queue.put(None)` no loop.
- **Por que `async`:** `run_coroutine_threadsafe` exige uma corrotina, e é ela
  que faz o `append` e a troca do `asyncio.Event` (que não é thread-safe)
  rodarem na thread do loop. Tirar o `async` faria essas chamadas levantarem
  `TypeError` e, se trocadas por chamada direta, mexeria no `Event` de fora do
  loop. É o mesmo formato de `asyncio.Queue.put`, que `RunEventLog`
  substituiu (ADR-159).
- **Testes:** `tests/gui/test_run_events.py` e os testes de
  `_execute_colab` em `tests/gui/test_colab_jobs.py` (eventos entregues antes
  de fechar o stream).
