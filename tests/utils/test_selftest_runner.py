"""Self-test runner: HTTP helpers, SSE tally, case verdicts and the report table.

The runner talks to a stand-in HTTP server on a loopback ephemeral port (no
training, no GPU, no network beyond 127.0.0.1).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from visionforge.utils import selftest
from visionforge.utils.selftest import (
    SelfTestCase,
    SelfTestOutcome,
    _EventCollector,
    _get,
    _post,
    _summarize,
    _validate_events,
    format_report,
    run_case,
    run_selftest,
)

RUN_ID = "r1"


class _Fake:
    """Scripted responses for the endpoints the runner touches."""

    def __init__(self) -> None:
        self.post_status = 200
        self.post_body: Any = {"run_id": RUN_ID}
        self.states: list[dict[str, Any]] = [{"status": "completed"}]
        self.report: dict[str, Any] = {"metrics": {"accuracy": 0.5}}
        self.sse: list[bytes] = [b'data: {"event": "epoch_end"}\n\n']
        self.posted: list[tuple[str, Any]] = []


def _handler(fake: _Fake) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def _send(
            self, code: int, body: bytes, ctype: str = "application/json"
        ) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            fake.posted.append((self.path, json.loads(self.rfile.read(length))))
            body = fake.post_body
            raw = body if isinstance(body, bytes) else json.dumps(body).encode()
            self._send(fake.post_status, raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/experiment/status":
                state = fake.states.pop(0) if len(fake.states) > 1 else fake.states[0]
                self._send(200, json.dumps(state).encode())
            elif self.path.startswith("/api/experiment/result/"):
                self._send(200, json.dumps({"report": fake.report}).encode())
            elif self.path == "/api/experiment/events":
                self._send(200, b"".join(fake.sse), "text/event-stream")
            else:
                self._send(404, b"{}")

    return Handler


@pytest.fixture
def server() -> Iterator[tuple[str, _Fake]]:
    fake = _Fake()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(fake))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", fake
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _case(**kw: Any) -> SelfTestCase:
    base: dict[str, Any] = {
        "task": "classification",
        "strategy": "simple",
        "endpoint": "/api/experiment/run",
        "payload": {"a": 1},
        "expect_keys": ("metrics",),
    }
    base.update(kw)
    return SelfTestCase(**base)


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selftest.time, "sleep", lambda _s: None)


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def test_post_sends_json_and_returns_status_and_body(
    server: tuple[str, _Fake],
) -> None:
    base, fake = server
    status, body = _post(base, "/x", {"k": [1, 2]})
    assert (status, body) == (200, {"run_id": RUN_ID})
    assert fake.posted == [("/x", {"k": [1, 2]})]


def test_post_http_error_returns_code_and_raw_text(server: tuple[str, _Fake]) -> None:
    base, fake = server
    fake.post_status = 422
    fake.post_body = b"bad payload"
    assert _post(base, "/x", {}) == (422, "bad payload")


def test_get_parses_json(server: tuple[str, _Fake]) -> None:
    base, _ = server
    assert _get(base, "/api/experiment/status") == {"status": "completed"}


# ── SSE tally ─────────────────────────────────────────────────────────────────


def test_event_collector_counts_kinds_and_skips_noise(
    server: tuple[str, _Fake],
) -> None:
    base, fake = server
    fake.sse = [
        b": keepalive\n",
        b'data: {"event": "epoch_end"}\n',
        b'data: {"event": "epoch_end"}\n',
        b'data: {"event": "trial_start"}\n',
        b"data: {}\n",
        b"data: \n",
        b'data: {"no_event": 1}\n',
    ]
    collector = _EventCollector(base)
    collector.start()
    assert collector.join(5) == {"epoch_end": 2, "trial_start": 1, "?": 1}


def test_event_collector_records_a_dead_stream() -> None:
    # nothing listens on port 1: the stream error is a finding, not a crash
    collector = _EventCollector("http://127.0.0.1:1")
    collector.start()
    assert collector.join(5) == {"<stream-error>": 1}


# ── event contract ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("counts", "multi", "expected"),
    [
        ({"<stream-error>": 1, "epoch_end": 3}, False, "SSE stream errored"),
        ({"epoch_end": 1}, False, ""),
        ({}, False, "no epoch_end events (got nothing)"),
        ({"trial_start": 2}, False, "no epoch_end events (got {'trial_start': 2})"),
        ({"trial_start": 1}, True, ""),
        ({}, True, "no trial_start events (got nothing)"),
        ({"epoch_end": 4}, True, "no trial_start events (got {'epoch_end': 4})"),
    ],
)
def test_validate_events(counts: dict[str, int], multi: bool, expected: str) -> None:
    assert _validate_events(counts, _case(multi_trial=multi)) == expected


# ── run_case verdicts ─────────────────────────────────────────────────────────


def test_run_case_passes_on_a_complete_run(server: tuple[str, _Fake]) -> None:
    base, fake = server
    fake.states = [{"status": "running"}, {"status": "completed"}]
    out = run_case(base, _case())
    assert out.status == "passed"
    assert out.detail == "accuracy=0.5000"
    assert out.events == {"epoch_end": 1}
    assert out.duration_s >= 0.0
    assert fake.posted == [("/api/experiment/run", {"a": 1})]


def test_run_case_reports_a_rejected_post(server: tuple[str, _Fake]) -> None:
    base, fake = server
    fake.post_status = 400
    fake.post_body = b"nope"
    out = run_case(base, _case())
    assert out.status == "failed"
    assert out.detail == "POST /api/experiment/run → 400: nope"


def test_run_case_reports_a_failed_run(server: tuple[str, _Fake]) -> None:
    base, fake = server
    fake.states = [{"status": "failed", "error": "boom"}]
    out = run_case(base, _case())
    assert (out.status, out.detail) == ("failed", "run failed: boom")


def test_run_case_times_out_when_the_run_never_ends(
    server: tuple[str, _Fake], monkeypatch: pytest.MonkeyPatch
) -> None:
    base, fake = server
    fake.states = [{"status": "running"}]
    monkeypatch.setattr(selftest, "CASE_TIMEOUT_S", 0.0)
    out = run_case(base, _case())
    assert out.status == "failed"
    assert out.detail == "timed out after 0s"


def test_run_case_reports_missing_report_keys(server: tuple[str, _Fake]) -> None:
    base, fake = server
    fake.report = {"zeta": 1}
    out = run_case(base, _case(expect_keys=("metrics", "headline")))
    assert out.status == "failed"
    assert out.detail == "report missing ['metrics', 'headline'] (got ['zeta'])"


def test_run_case_fails_when_the_event_contract_is_broken(
    server: tuple[str, _Fake],
) -> None:
    base, fake = server
    fake.sse = []
    out = run_case(base, _case())
    assert out.status == "failed"
    assert out.detail == "no epoch_end events (got nothing)"


def test_run_case_turns_an_unexpected_exception_into_a_failure(
    server: tuple[str, _Fake],
) -> None:
    base, fake = server
    fake.post_body = {"no_run_id": True}
    out = run_case(base, _case())
    assert out.status == "failed"
    assert out.detail == "KeyError: 'run_id'"
    assert out.duration_s >= 0.0


# ── summaries ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (
            {
                "kind": "replicated_comparison",
                "comparisons": [1, 2, 3],
                "significant_pairs": 2,
                "best_by_mean": "a",
                "underpowered": True,
            },
            "best=a 2/3 signif. (underpowered)",
        ),
        (
            {"kind": "replicated_comparison", "best_by_mean": "b"},
            "best=b 0/0 signif.",
        ),
        (
            {"metric": "mae", "headline": {"mean": 1.0, "ci95_high": 1.5}},
            "mae=1.0000±0.5000",
        ),
        ({"best_trial": {"metrics": {"f1": 0.25}}}, "best f1=0.2500"),
        ({"best_trial": {"metrics": {}}}, "best trial ok"),
        ({"aggregate": {"acc": {"mean": 0.75}}}, "acc=0.7500"),
        ({"aggregate": {}}, "ok"),
        ({"metrics": {"loss": 2.0}}, "loss=2.0000"),
        ({"metrics": {}}, "ok"),
        ({"eval": {"accuracy": 0.9}}, "accuracy=0.9000"),
        ({"mean_accuracy": 0.6}, "mean_accuracy=0.6000"),
        ({"test": {"name": "x", "iou": 0.5}}, "test.iou=0.5000"),
        ({"test": {"name": "x"}, "train": {"loss": 1}}, "train.loss=1.0000"),
        ({"detection": {"map": 0.125}}, "detection.map=0.1250"),
        ({"unrelated": 1}, "ok"),
    ],
)
def test_summarize(report: dict[str, Any], expected: str) -> None:
    assert _summarize(report) == expected


# ── run_selftest orchestration ────────────────────────────────────────────────


def test_run_selftest_runs_every_case_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, str, str]] = []

    @contextmanager
    def fake_serve() -> Iterator[str]:
        yield "http://stub"

    def fake_run_case(base: str, case: SelfTestCase) -> SelfTestOutcome:
        seen.append((base, case.task, case.strategy))
        status = "failed" if case.strategy == "cv" else "passed"
        return SelfTestOutcome(case.task, case.strategy, status=status, detail="d")

    monkeypatch.setattr(selftest, "serve_app", fake_serve)
    monkeypatch.setattr(selftest, "run_case", fake_run_case)
    workdir = tmp_path / "nested" / "work"
    outcomes = run_selftest(
        workdir, tasks=("classification",), strategies=("simple", "cv")
    )
    assert workdir.is_dir()
    assert seen == [
        ("http://stub", "classification", "simple"),
        ("http://stub", "classification", "cv"),
    ]
    assert [o.status for o in outcomes] == ["passed", "failed"]


# ── report table ──────────────────────────────────────────────────────────────


def test_format_report_empty() -> None:
    assert format_report([]) == "No self-test cases selected."


def test_format_report_table_and_verdict() -> None:
    outcomes = [
        SelfTestOutcome("classification", "simple", "passed", 1.5, "acc=1"),
        SelfTestOutcome("detection", "cv", "failed", 2.0, "x" * 100),
        SelfTestOutcome("anomaly", "sweep", "skipped", 0.0, ""),
        SelfTestOutcome("anomaly", "cv", "weird", 0.0, ""),
    ]
    lines = format_report(outcomes).splitlines()
    assert lines[0].startswith("case") and "detail" in lines[0]
    assert "PASS" in lines[2] and "classification/simple" in lines[2]
    assert "1.5s" in lines[2]
    assert "FAIL" in lines[3]
    assert lines[3].endswith("x" * 60) and "x" * 61 not in lines[3]
    assert "SKIP" in lines[4]
    assert lines[5].split()[1] == "?"
    assert lines[-1] == "1/4 cases passed — failed: detection/cv"


def test_format_report_all_passed_has_no_failed_suffix() -> None:
    out = format_report([SelfTestOutcome("a", "b", "passed")])
    assert out.splitlines()[-1] == "1/1 cases passed"
