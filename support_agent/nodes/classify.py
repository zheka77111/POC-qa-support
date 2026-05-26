from __future__ import annotations

from support_agent.logging_utils import append_error, append_event
from support_agent.prompts import build_classification_prompt
from support_agent.state import SupportTicketState


def receive_request(state: SupportTicketState) -> SupportTicketState:
    state["current_node"] = "ReceiveRequest"
    state.setdefault("events", [])
    state.setdefault("errors", [])
    state.setdefault("retrieved_docs", [])
    state.setdefault("refinement_count", 0)
    state.setdefault("escalated", False)
    state.setdefault("escalation_reason", None)
    state.setdefault("has_error", False)
    append_event(state, "ReceiveRequest", "request_received", {"ticket_id": state.get("ticket_id")})
    return state


def classify_request(state: SupportTicketState, llm_client) -> SupportTicketState:
    state["current_node"] = "ClassifyRequest"
    try:
        system_prompt, user_prompt = build_classification_prompt(state)
        data = llm_client.invoke_json(system_prompt, user_prompt)
        state["is_complaint"] = bool(data.get("is_complaint", False))
        state["category"] = str(data.get("category", "other"))
        state["sentiment"] = str(data.get("sentiment", "neutral"))
        state["urgency"] = str(data.get("urgency", "medium"))
        append_event(
            state,
            "ClassifyRequest",
            "classification_done",
            {
                "is_complaint": state["is_complaint"],
                "category": state["category"],
                "sentiment": state["sentiment"],
                "urgency": state["urgency"],
            },
        )
        return state
    except Exception as exc:
        state["has_error"] = True
        state["error_node"] = "ClassifyRequest"
        append_error(state, "ClassifyRequest", type(exc).__name__, str(exc))
        append_event(state, "ClassifyRequest", "classification_failed", {"error": str(exc)})
        return state

