from __future__ import annotations

from support_agent.logging_utils import make_error, make_event
from support_agent.state import SupportTicketState


def prepare_search_context(state: SupportTicketState) -> SupportTicketState:
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
        search_filters = {"domain": domain} if domain else {}
        extracted_entities = {
            "error_code": "500" if "500" in text else None,
            "mentions_payment": any(x in text.lower() for x in ["оплат", "карт", "billing"]),
            "mentions_account": any(x in text.lower() for x in ["аккаунт", "кабинет", "парол"]),
        }
        return {
            "current_node": "PrepareSearchContext",
            "search_query": text,
            "search_filters": search_filters,
            "extracted_entities": extracted_entities,
            "events": [
                make_event(
                    state,
                    "PrepareSearchContext",
                    "search_context_prepared",
                    {
                        "search_filters": search_filters,
                        "extracted_entities": extracted_entities,
                    },
                )
            ],
        }
    except Exception as exc:
        return {
            "current_node": "PrepareSearchContext",
            "has_error": True,
            "error_node": "PrepareSearchContext",
            "errors": [make_error(state, "PrepareSearchContext", type(exc).__name__, str(exc))],
            "events": [
                make_event(
                    state,
                    "PrepareSearchContext",
                    "search_context_failed",
                    {"error": str(exc)},
                )
            ],
        }


def search_knowledge_base(state: SupportTicketState, kb, top_k: int) -> SupportTicketState:
    try:
        docs = kb.search(
            query=state.get("search_query", state.get("user_text", "")),
            filters=state.get("search_filters", {}),
            top_k=top_k,
        )
        return {
            "current_node": "SearchKnowledgeBase",
            "retrieved_docs": docs,
            "events": [
                make_event(
                    state,
                    "SearchKnowledgeBase",
                    "kb_search_completed",
                    {"docs_found": len(docs)},
                )
            ],
        }
    except Exception as exc:
        return {
            "current_node": "SearchKnowledgeBase",
            "has_error": True,
            "error_node": "SearchKnowledgeBase",
            "errors": [make_error(state, "SearchKnowledgeBase", type(exc).__name__, str(exc))],
            "events": [
                make_event(
                    state,
                    "SearchKnowledgeBase",
                    "kb_search_failed",
                    {"error": str(exc)},
                )
            ],
        }
