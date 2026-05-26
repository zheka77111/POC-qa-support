from __future__ import annotations

from support_agent.logging_utils import append_error, append_event
from support_agent.prompts import build_evaluation_prompt
from support_agent.state import SupportTicketState


def evaluate_answer(state: SupportTicketState, llm_client) -> SupportTicketState:
    state["current_node"] = "EvaluateAnswer"
    try:
        system_prompt, user_prompt = build_evaluation_prompt(state)
        data = llm_client.invoke_json(system_prompt, user_prompt)
        completeness = float(data.get("completeness", 0.0))
        politeness = float(data.get("politeness", 0.0))
        relevance = float(data.get("relevance", 0.0))
        notes = str(data.get("notes", ""))

        quality_score = 0.4 * completeness + 0.2 * politeness + 0.4 * relevance

        state["completeness"] = max(0.0, min(1.0, completeness))
        state["politeness"] = max(0.0, min(1.0, politeness))
        state["relevance"] = max(0.0, min(1.0, relevance))
        state["quality_score"] = round(max(0.0, min(1.0, quality_score)), 3)
        state["evaluation_notes"] = notes

        append_event(
            state,
            "EvaluateAnswer",
            "quality_evaluated",
            {
                "completeness": state["completeness"],
                "politeness": state["politeness"],
                "relevance": state["relevance"],
                "quality_score": state["quality_score"],
            },
        )
        return state
    except Exception as exc:
        state["has_error"] = True
        state["error_node"] = "EvaluateAnswer"
        append_error(state, "EvaluateAnswer", type(exc).__name__, str(exc))
        append_event(state, "EvaluateAnswer", "quality_evaluation_failed", {"error": str(exc)})
        return state

