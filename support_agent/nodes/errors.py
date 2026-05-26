from __future__ import annotations

from support_agent.logging_utils import append_event
from support_agent.state import SupportTicketState


def handle_error(state: SupportTicketState) -> SupportTicketState:
    state["current_node"] = "HandleError"
    append_event(
        state,
        "HandleError",
        "error_handled",
        {
            "error_node": state.get("error_node"),
            "errors_count": len(state.get("errors", [])),
        },
    )
    return state

