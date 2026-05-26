from __future__ import annotations

from support_agent.logging_utils import append_event
from support_agent.state import SupportTicketState


def escalate_complaint(state: SupportTicketState) -> SupportTicketState:
    state["current_node"] = "EscalateComplaint"
    state["escalated"] = True
    state["escalation_reason"] = "complaint_detected"
    state["final_response"] = (
        "Ваше обращение передано оператору. Специалист свяжется с вами для детального рассмотрения ситуации."
    )
    append_event(state, "EscalateComplaint", "escalated", {"reason": state["escalation_reason"]})
    return state


def escalate_low_quality(state: SupportTicketState) -> SupportTicketState:
    state["current_node"] = "EscalateLowQuality"
    state["escalated"] = True
    state["escalation_reason"] = "low_quality_answer"
    state["final_response"] = (
        "В базе знаний недостаточно информации для точного ответа. "
        "Передаю обращение специалисту для дополнительной проверки."
    )
    append_event(state, "EscalateLowQuality", "escalated", {"reason": state["escalation_reason"]})
    return state


def escalate_technical_error(state: SupportTicketState) -> SupportTicketState:
    state["current_node"] = "EscalateTechnicalError"
    state["escalated"] = True
    state["escalation_reason"] = "technical_error"
    state["final_response"] = (
        "Сейчас возникла техническая ошибка при обработке обращения. "
        "Мы передали запрос оператору и вернемся с ответом."
    )
    append_event(state, "EscalateTechnicalError", "escalated", {"reason": state["escalation_reason"]})
    return state

