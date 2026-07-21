"""Generate the read-only calendar-coverage audit for official MetroPT-3."""

from __future__ import annotations

from common import audit_metropt3_calendar


if __name__ == "__main__":
    _, summary = audit_metropt3_calendar()
    print(summary)
