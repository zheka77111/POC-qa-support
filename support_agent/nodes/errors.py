from __future__ import annotations

from support_agent.logging_utils import make_event
from support_agent.state import SupportTicketState


def handle_error(state: SupportTicketState) -> SupportTicketState:
    return {
        "current_node": "HandleError",
        "events": [
            make_event(
                state,
                "HandleError",
                "error_handled",
                {
                    "error_node": state.get("error_node"),
                    "errors_count": len(state.get("errors", [])),
                },
            )
        ],
    }
