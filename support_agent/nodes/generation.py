from __future__ import annotations

from support_agent.llm import LLMClient
from support_agent.logging_utils import make_error, make_event
from support_agent.prompts import build_generation_prompt, build_refinement_prompt
from support_agent.state import SupportTicketState


def generate_answer(state: SupportTicketState, llm_client: LLMClient) -> SupportTicketState:
    try:
        system_prompt, user_prompt = build_generation_prompt(state)
        draft_response = llm_client.invoke_text(system_prompt, user_prompt)
        return {
            "current_node": "GenerateAnswer",
            "draft_response": draft_response,
            "events": [make_event(state, "GenerateAnswer", "answer_generated")],
        }
    except Exception as exc:
        return {
            "current_node": "GenerateAnswer",
            "has_error": True,
            "error_node": "GenerateAnswer",
            "errors": [make_error(state, "GenerateAnswer", type(exc).__name__, str(exc))],
            "events": [
                make_event(
                    state,
                    "GenerateAnswer",
                    "answer_generation_failed",
                    {"error": str(exc)},
                )
            ],
        }


def generate_fallback_answer(state: SupportTicketState) -> SupportTicketState:
    draft_response = (
        "Спасибо за вопрос. Сейчас мне не хватает подтвержденной информации в базе знаний, "
        "чтобы дать точный ответ. Уточните детали, пожалуйста, или мы передадим запрос специалисту."
    )
    return {
        "current_node": "GenerateFallbackAnswer",
        "draft_response": draft_response,
        "events": [make_event(state, "GenerateFallbackAnswer", "fallback_answer_generated")],
    }


def refine_answer(state: SupportTicketState, llm_client: LLMClient) -> SupportTicketState:
    try:
        system_prompt, user_prompt = build_refinement_prompt(state)
        draft_response = llm_client.invoke_text(system_prompt, user_prompt)
        refinement_count = state.get("refinement_count", 0) + 1
        return {
            "current_node": "RefineAnswer",
            "draft_response": draft_response,
            "refinement_count": refinement_count,
            "events": [
                make_event(
                    state,
                    "RefineAnswer",
                    "answer_refined",
                    {"refinement_count": refinement_count},
                )
            ],
        }
    except Exception as exc:
        return {
            "current_node": "RefineAnswer",
            "has_error": True,
            "error_node": "RefineAnswer",
            "errors": [make_error(state, "RefineAnswer", type(exc).__name__, str(exc))],
            "events": [
                make_event(
                    state,
                    "RefineAnswer",
                    "refinement_failed",
                    {"error": str(exc)},
                )
            ],
        }


def return_answer(state: SupportTicketState) -> SupportTicketState:
    return {
        "current_node": "ReturnAnswer",
        "final_response": state.get("draft_response", ""),
        "escalated": False,
        "escalation_reason": None,
        "events": [make_event(state, "ReturnAnswer", "completed")],
    }
