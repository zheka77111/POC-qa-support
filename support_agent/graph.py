from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from support_agent.config import Settings
from support_agent.knowledge_base import HybridChromaKnowledgeBase
from support_agent.llm import build_llm_client
from support_agent.nodes.classify import classify_request, receive_request
from support_agent.nodes.errors import handle_error
from support_agent.nodes.escalation import (
    escalate_complaint,
    escalate_low_quality,
    escalate_technical_error,
)
from support_agent.nodes.evaluation import evaluate_answer
from support_agent.nodes.generation import (
    generate_answer,
    generate_fallback_answer,
    refine_answer,
    return_answer,
)
from support_agent.nodes.retrieval import prepare_search_context, search_knowledge_base
from support_agent.state import SupportTicketState


def build_support_graph(
    settings: Settings,
    logger: logging.Logger,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    llm_client_gigachat = build_llm_client(settings)
    kb_files = settings.files
    kb = HybridChromaKnowledgeBase.from_markdown_files([Path(f) for f in kb_files], settings=settings)

    graph = StateGraph(SupportTicketState)

    graph.add_node("ReceiveRequest", receive_request)
    graph.add_node("ClassifyRequest", partial(classify_request, llm_client=llm_client_gigachat))
    graph.add_node("EscalateComplaint", escalate_complaint)
    graph.add_node("PrepareSearchContext", prepare_search_context)
    graph.add_node(
        "SearchKnowledgeBase",
        partial(search_knowledge_base, kb=kb, top_k=settings.kb_top_k),
    )
    graph.add_node("GenerateAnswer", partial(generate_answer, llm_client=llm_client_gigachat))
    graph.add_node("GenerateFallbackAnswer", generate_fallback_answer)
    graph.add_node("EvaluateAnswer", partial(evaluate_answer, llm_client=llm_client_gigachat))
    graph.add_node("RefineAnswer", partial(refine_answer, llm_client=llm_client_gigachat))
    graph.add_node("ReturnAnswer", return_answer)
    graph.add_node("EscalateLowQuality", escalate_low_quality)
    graph.add_node("HandleError", handle_error)
    graph.add_node("EscalateTechnicalError", escalate_technical_error)

    graph.set_entry_point("ReceiveRequest")
    graph.add_edge("ReceiveRequest", "ClassifyRequest")

    graph.add_conditional_edges(
        "ClassifyRequest",
        route_after_classification,
        {
            "error": "HandleError",
            "complaint": "EscalateComplaint",
            "non_complaint": "PrepareSearchContext",
        },
    )
    graph.add_conditional_edges(
        "PrepareSearchContext",
        route_on_error_or_continue,
        {"error": "HandleError", "ok": "SearchKnowledgeBase"},
    )
    graph.add_conditional_edges(
        "SearchKnowledgeBase",
        route_after_search,
        {"error": "HandleError", "has_docs": "GenerateAnswer", "no_docs": "GenerateFallbackAnswer"},
    )
    graph.add_conditional_edges(
        "GenerateAnswer",
        route_on_error_or_continue,
        {"error": "HandleError", "ok": "EvaluateAnswer"},
    )
    graph.add_edge("GenerateFallbackAnswer", "EvaluateAnswer")
    graph.add_conditional_edges(
        "EvaluateAnswer",
        route_after_evaluation,
        {
            "error": "HandleError",
            "accept": "ReturnAnswer",
            "refine": "RefineAnswer",
            "escalate": "EscalateLowQuality",
        },
    )
    graph.add_conditional_edges(
        "RefineAnswer",
        route_on_error_or_continue,
        {"error": "HandleError", "ok": "EvaluateAnswer"},
    )

    graph.add_edge("HandleError", "EscalateTechnicalError")

    graph.add_edge("EscalateComplaint", END)
    graph.add_edge("EscalateLowQuality", END)
    graph.add_edge("EscalateTechnicalError", END)
    graph.add_edge("ReturnAnswer", END)

    app = graph.compile(checkpointer=checkpointer)
    return app


def route_on_error_or_continue(state: SupportTicketState) -> str:
    if state.get("has_error"):
        return "error"
    return "ok"


def route_after_classification(state: SupportTicketState) -> str:
    if state.get("has_error"):
        return "error"
    if state.get("is_complaint"):
        return "complaint"
    return "non_complaint"


def route_after_search(state: SupportTicketState) -> str:
    if state.get("has_error"):
        return "error"
    return "has_docs" if state.get("retrieved_docs") else "no_docs"


def route_after_evaluation(state: SupportTicketState) -> str:
    if state.get("has_error"):
        return "error"
    if state.get("quality_score", 0.0) >= state.get("quality_threshold", 0.75):
        return "accept"
    if state.get("refinement_count", 0) < state.get("max_refinements", 2):
        return "refine"
    return "escalate"
