from __future__ import annotations

from support_agent.logging_utils import make_event
from support_agent.state import SupportTicketState


def escalate_complaint(state: SupportTicketState) -> SupportTicketState:
    escalation_reason = "complaint_detected"
    final_response = (
        "Ваше обращение передано оператору. Специалист свяжется с вами для детального рассмотрения ситуации."
    )
    return {
        "current_node": "EscalateComplaint",
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


def escalate_low_quality(state: SupportTicketState) -> SupportTicketState:
    escalation_reason = "low_quality_answer"
    final_response = (
        "В базе знаний недостаточно информации для точного ответа. "
        "Передаю обращение специалисту для дополнительной проверки."
    )
    return {
        "current_node": "EscalateLowQuality",
        "escalated": True,
        "escalation_reason": escalation_reason,
        "final_response": final_response,
        "events": [
            make_event(
                state,
                "EscalateLowQuality",
                "escalated",
                {"reason": escalation_reason},
            )
        ],
    }


def escalate_technical_error(state: SupportTicketState) -> SupportTicketState:
    escalation_reason = "technical_error"
    final_response = (
        "Сейчас возникла техническая ошибка при обработке обращения. "
        "Мы передали запрос оператору и вернемся с ответом."
    )
    return {
        "current_node": "EscalateTechnicalError",
        "escalated": True,
        "escalation_reason": escalation_reason,
        "final_response": final_response,
        "events": [
            make_event(
                state,
                "EscalateTechnicalError",
                "escalated",
                {"reason": escalation_reason},
            )
        ],
    }
