from __future__ import annotations

import argparse
import json
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from loguru import Logger
from support_agent.config import Settings, configure_langsmith
from support_agent.graph import build_support_graph
from support_agent.knowledge_base import HybridChromaKnowledgeBase
from support_agent.logging_utils import setup_logger
from support_agent.state import SupportTicketState
from support_agent.test_cases import TEST_CASES_50


def run_single(
    app: CompiledStateGraph,
    settings: Settings,
    logger: Logger,
    ticket_id: str,
    user_text: str,
    threshold: float | None = None,
    max_model_retries: int | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "ticket_id": ticket_id,
        "user_text": user_text,
        "messages": [HumanMessage(content=user_text)],
        "quality_threshold": threshold if threshold is not None else settings.quality_threshold,
        "max_model_retries": (
            max_model_retries if max_model_retries is not None else settings.max_model_retries
        ),
    }
    config = RunnableConfig(configurable={"thread_id": ticket_id})
    result = app.invoke(state, config=config)
    logger.info(f"Final trace: {app.get_state(config=config)}")
    return {
        "ticket_id": ticket_id,
        "final_response": result.get("final_response", ""),
        "escalated": bool(result.get("escalated", False)),
        "escalation_reason": result.get("escalation_reason"),
        "category": result.get("category"),
        "quality_score": result.get("quality_score"),
        "events_count": len(result.get("events", [])),
        "errors_count": len(result.get("errors", [])),
    }


def run_batch(app: CompiledStateGraph, settings: Settings, logger: Logger) -> dict[str, Any]:
    total = len(TEST_CASES_50)
    exact_expected = 0
    escalated_count = 0
    results = []

    for case in TEST_CASES_50:
        case_data = cast(dict[str, Any], case)
        out = run_single(app, settings, logger, case_data["ticket_id"], case_data["user_text"])
        expected = case_data["expected"]
        if expected == "complaint_escalation":
            ok = out["escalated"] is True
        else:
            ok = out["escalated"] is False
        if ok:
            exact_expected += 1
        if out["escalated"]:
            escalated_count += 1
        results.append(
            {
                "ticket_id": case_data["ticket_id"],
                "expected": expected,
                "actual_escalated": out["escalated"],
                "actual_reason": out["escalation_reason"],
                "quality_score": out["quality_score"],
                "ok": ok,
            }
        )

    return {
        "total": total,
        "matched_expected": exact_expected,
        "accuracy": round(exact_expected / total, 3) if total else 0.0,
        "escalated_count": escalated_count,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Support ticket LangGraph PoC")
    parser.add_argument("--ticket-id", type=str, default="TICKET-LOCAL-001")
    parser.add_argument("--text", type=str, default="Не могу войти в личный кабинет, забыл пароль.")
    parser.add_argument("--batch", action="store_true", help="Run all test cases")
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--max-model-retries", type=int, default=None)
    args = parser.parse_args()

    settings = Settings.from_env()
    configure_langsmith(settings)
    logger = setup_logger(settings.log_level, log_file = f"logs/support_agent_{args.ticket_id}.log")
    checkpointer = MemorySaver()
    kb = HybridChromaKnowledgeBase.from_markdown_files(settings=settings)
    app = build_support_graph(
        settings=settings,
        kb=kb,
        checkpointer=checkpointer,
    )

    if args.batch:
        summary = run_batch(app, settings, logger)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    output = run_single(
        app,
        settings,
        logger,
        ticket_id=args.ticket_id,
        user_text=args.text,
        threshold=args.quality_threshold,
        max_model_retries=args.max_model_retries,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
