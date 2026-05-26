from __future__ import annotations

from support_agent.logging_utils import append_error, append_event
from support_agent.state import SupportTicketState


def prepare_search_context(state: SupportTicketState) -> SupportTicketState:
    state["current_node"] = "PrepareSearchContext"
    try:
        text = state.get("user_text", "")
        category = state.get("category", "other")
        domain_map = {
            "billing_question": "billing",
            "technical_question": "technical",
            "how_to": "how_to",
            "complaint": "technical",
            "other": None,
        }
        domain = domain_map.get(category)
        state["search_query"] = text
        state["search_filters"] = {"domain": domain} if domain else {}
        state["extracted_entities"] = {
            "error_code": "500" if "500" in text else None,
            "mentions_payment": any(x in text.lower() for x in ["оплат", "карт", "billing"]),
            "mentions_account": any(x in text.lower() for x in ["аккаунт", "кабинет", "парол"]),
        }
        append_event(
            state,
            "PrepareSearchContext",
            "search_context_prepared",
            {
                "search_filters": state["search_filters"],
                "extracted_entities": state["extracted_entities"],
            },
        )
        return state
    except Exception as exc:
        state["has_error"] = True
        state["error_node"] = "PrepareSearchContext"
        append_error(state, "PrepareSearchContext", type(exc).__name__, str(exc))
        append_event(state, "PrepareSearchContext", "search_context_failed", {"error": str(exc)})
        return state


def search_knowledge_base(state: SupportTicketState, kb, top_k: int) -> SupportTicketState:
    state["current_node"] = "SearchKnowledgeBase"
    try:
        docs = kb.search(
            query=state.get("search_query", state.get("user_text", "")),
            filters=state.get("search_filters", {}),
            top_k=top_k,
        )
        state["retrieved_docs"] = docs
        append_event(
            state,
            "SearchKnowledgeBase",
            "kb_search_completed",
            {"docs_found": len(docs)},
        )
        return state
    except Exception as exc:
        state["has_error"] = True
        state["error_node"] = "SearchKnowledgeBase"
        append_error(state, "SearchKnowledgeBase", type(exc).__name__, str(exc))
        append_event(state, "SearchKnowledgeBase", "kb_search_failed", {"error": str(exc)})
        return state

