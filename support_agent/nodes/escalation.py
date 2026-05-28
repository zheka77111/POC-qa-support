from __future__ import annotations

from typing import Any
import json
from support_agent.logging_utils import make_event
from support_agent.state import SupportTicketState


def escalate_complaint(state: SupportTicketState) -> dict[str, Any]:

    

    escalation_reason = json.dumps(
                {
                    "reason": "complaint_detected",
                    "user_query": state.get("user_text", ""),
                    "category": state.get("category", ""),
                    "urgency": state.get("urgency", ""),
                },
                ensure_ascii=False,
            )
    final_response = (
        "Ваше обращение передано оператору. Специалист свяжется с вами для детального рассмотрения ситуации."
    )
    return {
        "escalated": True,
        "escalation_reason": escalation_reason,
        "final_response": final_response,
        "events": [
            make_event(
                state,
                "EscalateComplaint",
                "escalated",
                {"reason": escalation_reason},
            )
        ],
    }

