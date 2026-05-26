from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from support_agent.state import SupportTicketState


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("support_agent")
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_event(
    state: SupportTicketState,
    node: str,
    event: str,
    payload: dict[str, Any] | None = None,
) -> None:
    state.setdefault("events", [])
    state["events"].append(
        {
            "timestamp": now_iso(),
            "ticket_id": state.get("ticket_id"),
            "node": node,
            "event": event,
            "payload": payload or {},
        }
    )


def append_error(
    state: SupportTicketState,
    node: str,
    error_type: str,
    message: str,
) -> None:
    state.setdefault("errors", [])
    state["errors"].append(
        {
            "timestamp": now_iso(),
            "ticket_id": state.get("ticket_id"),
            "node": node,
            "error_type": error_type,
            "message": message,
        }
    )

