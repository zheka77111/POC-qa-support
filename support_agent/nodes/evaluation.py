from __future__ import annotations

from support_agent.llm import AnswerEvaluation, LLMClient
from support_agent.logging_utils import make_error, make_event
from support_agent.prompts import build_evaluation_prompt
from support_agent.state import SupportTicketState


def evaluate_answer(state: SupportTicketState, llm_client: LLMClient) -> SupportTicketState:
    try:
        system_prompt, user_prompt = build_evaluation_prompt(state)
        data: AnswerEvaluation = llm_client.invoke_structured(
            system_prompt,
            user_prompt,
            AnswerEvaluation,
        )
        completeness = data.completeness
        politeness = data.politeness
        relevance = data.relevance
        notes = data.notes

        quality_score = 0.4 * completeness + 0.2 * politeness + 0.4 * relevance
        quality_score = round(max(0.0, min(1.0, quality_score)), 3)

        return {
            "current_node": "EvaluateAnswer",
            "completeness": completeness,
            "politeness": politeness,
            "relevance": relevance,
            "quality_score": quality_score,
            "evaluation_notes": notes,
            "events": [
                make_event(
                    state,
                    "EvaluateAnswer",
                    "quality_evaluated",
                    {
                        "completeness": completeness,
                        "politeness": politeness,
                        "relevance": relevance,
                        "quality_score": quality_score,
                    },
                )
            ],
        }
    except Exception as exc:
        return {
            "current_node": "EvaluateAnswer",
            "has_error": True,
            "error_node": "EvaluateAnswer",
            "errors": [make_error(state, "EvaluateAnswer", type(exc).__name__, str(exc))],
            "events": [
                make_event(
                    state,
                    "EvaluateAnswer",
                    "quality_evaluation_failed",
                    {"error": str(exc)},
                )
            ],
        }
