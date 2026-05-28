from __future__ import annotations
from pathlib import Path

from loguru import logger
from datetime import datetime, timezone
from typing import Any

from support_agent.state import SupportTicketState


def setup_logger(
    level: str = "INFO",
    log_file: str = "logs/support_agent.log",
): 
    logger.remove()

    # Консоль
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        colorize=True,
    )

    # Файл
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        sink=path,
        level=level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        encoding="utf-8",
    )

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


def make_event(
    state: SupportTicketState,
    node: str,
    event: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": now_iso(),
        "ticket_id": state.get("ticket_id"),
        "node": node,
        "event": event,
        "payload": payload or {},
    }


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


def make_error(
    state: SupportTicketState,
    node: str,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        "timestamp": now_iso(),
        "ticket_id": state.get("ticket_id"),
        "node": node,
        "error_type": error_type,
        "message": message,
    }

