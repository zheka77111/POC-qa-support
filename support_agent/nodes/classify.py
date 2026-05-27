from __future__ import annotations

from support_agent.llm import LLMClient, TicketClassification
from support_agent.logging_utils import make_error, make_event
from support_agent.prompts import build_classification_prompt
from support_agent.state import SupportTicketState


def receive_request(state: SupportTicketState) -> SupportTicketState:
    return {
        "current_node": "ReceiveRequest",
        "events": [
            make_event(
                state,
                "ReceiveRequest",
                "request_received",
                {"ticket_id": state.get("ticket_id")},
            )
        ],
        "retrieved_docs": state.get("retrieved_docs", []),
        "refinement_count": state.get("refinement_count", 0),
        "escalated": state.get("escalated", False),
        "escalation_reason": state.get("escalation_reason"),
        "has_error": state.get("has_error", False),
    }


def classify_request(state: SupportTicketState, llm_client: LLMClient) -> SupportTicketState:
    try:
        system_prompt, user_prompt = build_classification_prompt(state)
        data: TicketClassification = llm_client.invoke_structured(
            system_prompt,
            user_prompt,
            TicketClassification,
        )
        return {
            "current_node": "ClassifyRequest",
            "is_complaint": data.is_complaint,
            "category": data.category,
            "events": [
                make_event(
                    state,
                    "ClassifyRequest",
                    "classification_done",
                    {
                        "is_complaint": data.is_complaint,
                        "category": data.category,
                    },
                )
            ],
        }
    except Exception as exc:
        return {
            "current_node": "ClassifyRequest",
            "has_error": True,
            "error_node": "ClassifyRequest",
            "errors": [make_error(state, "ClassifyRequest", type(exc).__name__, str(exc))],
            "events": [
                make_event(
                    state,
                    "ClassifyRequest",
                    "classification_failed",
                    {"error": str(exc)},
                )
            ],
        }
