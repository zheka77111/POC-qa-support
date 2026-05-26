from __future__ import annotations

from support_agent.logging_utils import append_error, append_event
from support_agent.prompts import build_generation_prompt, build_refinement_prompt
from support_agent.state import SupportTicketState


def generate_answer(state: SupportTicketState, llm_client) -> SupportTicketState:
    state["current_node"] = "GenerateAnswer"
    try:
        system_prompt, user_prompt = build_generation_prompt(state)
        state["draft_response"] = llm_client.invoke_text(system_prompt, user_prompt)
        append_event(state, "GenerateAnswer", "answer_generated")
        return state
    except Exception as exc:
        state["has_error"] = True
        state["error_node"] = "GenerateAnswer"
        append_error(state, "GenerateAnswer", type(exc).__name__, str(exc))
        append_event(state, "GenerateAnswer", "answer_generation_failed", {"error": str(exc)})
        return state


def generate_fallback_answer(state: SupportTicketState) -> SupportTicketState:
    state["current_node"] = "GenerateFallbackAnswer"
    state["draft_response"] = (
        "Спасибо за вопрос. Сейчас мне не хватает подтвержденной информации в базе знаний, "
        "чтобы дать точный ответ. Уточните детали, пожалуйста, или мы передадим запрос специалисту."
    )
    append_event(state, "GenerateFallbackAnswer", "fallback_answer_generated")
    return state


def refine_answer(state: SupportTicketState, llm_client) -> SupportTicketState:
    state["current_node"] = "RefineAnswer"
    try:
        system_prompt, user_prompt = build_refinement_prompt(state)
        state["draft_response"] = llm_client.invoke_text(system_prompt, user_prompt)
        state["refinement_count"] = state.get("refinement_count", 0) + 1
        append_event(
            state,
            "RefineAnswer",
            "answer_refined",
            {"refinement_count": state["refinement_count"]},
        )
        return state
    except Exception as exc:
        state["has_error"] = True
        state["error_node"] = "RefineAnswer"
        append_error(state, "RefineAnswer", type(exc).__name__, str(exc))
        append_event(state, "RefineAnswer", "refinement_failed", {"error": str(exc)})
        return state


def return_answer(state: SupportTicketState) -> SupportTicketState:
    state["current_node"] = "ReturnAnswer"
    state["final_response"] = state.get("draft_response", "")
    state["escalated"] = False
    state["escalation_reason"] = None
    append_event(state, "ReturnAnswer", "completed")
    return state

