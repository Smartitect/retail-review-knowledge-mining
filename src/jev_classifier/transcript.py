"""
What crossed the wire to Jev, printed so you can read it.

Records go to a logger of their own, `jev_classifier.transcript`, one JSON
object per exchange. It is off unless something calls `log_to_stdout()`: a
null handler and no propagation mean a process that has not opted in never
builds a record, and a harness that sets the root logger to INFO does not
switch it on by accident.

Both sides are taken from the HTTP exchange itself - the request body the SDK
actually sent and the response body the API actually returned - rather than
rebuilt from our own objects, so the transcript cannot disagree with the wire.
Only when a call fails with no response is the request rebuilt from the state
and questions that were passed in.

Every request carries the same question set, which is most of its size, so by
default it is printed in full on the first exchange of a run and elided after
that. `log_to_stdout(full_questions=True)` prints it every time.

The API key is never printed: it travels in a request header, and headers are
not recorded.
"""

import json
import logging
import sys
from datetime import UTC, datetime

LOGGER_NAME = "jev_classifier.transcript"

log = logging.getLogger(LOGGER_NAME)
log.addHandler(logging.NullHandler())
log.propagate = False

INDENT = "  "

_options = {"full_questions": False, "questions_shown": False}


def log_to_stdout(stream=None, *, full_questions: bool = False, level: int = logging.INFO) -> logging.Handler:
    """Print every exchange to standard out. Calling it again replaces the handler rather than adding one."""
    silence()
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._jev_transcript = True
    log.addHandler(handler)
    log.setLevel(level)
    _options.update(full_questions=full_questions, questions_shown=False)
    return handler


def silence() -> None:
    """Stop printing: remove the handlers this module added."""
    for handler in list(log.handlers):
        if getattr(handler, "_jev_transcript", False):
            log.removeHandler(handler)
    log.setLevel(logging.NOTSET)


def enabled() -> bool:
    """Whether a handler of this module's own is listening."""
    return log.isEnabledFor(logging.INFO) and any(getattr(h, "_jev_transcript", False) for h in log.handlers)


def exchange(*, key: dict, sent: dict, received: dict | None, latency_ms: int,
             request_id: str | None = None, error: str | None = None) -> None:
    """Record one request and its response (or its error)."""
    if not enabled():
        return
    if not _options["full_questions"] and "questions" in sent:
        if _options["questions_shown"]:
            sent = {**sent, "questions": f"<elided: the same {len(sent['questions'])} questions as the first exchange>"}
        else:
            _options["questions_shown"] = True
    record = {
        "at": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "sentence": key,
        "request_id": request_id,
        "latency_ms": latency_ms,
        "sent": sent,
        "received": received,
    }
    if error is not None:
        record["error"] = error
    log.info(pretty(record))


def pretty(value, depth: int = 0) -> str:
    """JSON, with any list of plain values, and any small flat object, kept on one line.

    `json.dumps(indent=2)` gives every probability its own line, which turns one
    Choice answer into twenty. The result is still JSON: only whitespace differs.
    """
    pad, inner = INDENT * depth, INDENT * (depth + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        flat = all(not isinstance(v, (dict, list)) for v in value.values())
        one_line = json.dumps(value, default=str, ensure_ascii=False)
        if flat and len(one_line) <= 110:
            return one_line
        items = (f"{inner}{json.dumps(str(k))}: {pretty(v, depth + 1)}" for k, v in value.items())
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(not isinstance(item, (dict, list)) for item in value):
            return json.dumps(value, default=str, ensure_ascii=False)
        items = (f"{inner}{pretty(item, depth + 1)}" for item in value)
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(value, default=str, ensure_ascii=False)


__all__ = ["LOGGER_NAME", "enabled", "exchange", "log", "log_to_stdout", "pretty", "silence"]
