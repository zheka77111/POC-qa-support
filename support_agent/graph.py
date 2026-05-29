from __future__ import annotations

from typing import Annotated, Any, Callable, Literal, cast
from loguru import logger
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, Runtime, after_model, before_agent, wrap_model_call
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, ToolException
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from langgraph.types import Command
from support_agent.config import Settings
from support_agent.error import (
    ClassifyCategoryNodeError,
    ClassifyComplaintNodeError,
    ClassifyUrgencyNodeError,
    EscalateComplaintNodeError,
    FinalizeResponseNodeError,
    GatherClassificationsNodeError,
    KnowledgeBaseSearchToolError,
    ReceiveRequestNodeError,
)
from support_agent.knowledge_base import HybridChromaKnowledgeBase
from support_agent.logging_utils import make_error, make_event
from support_agent.nodes.escalation import escalate_complaint
from support_agent.prompts import (
    ANSWER_QUALITY_EVALUATION_SYSTEM_PROMPT,
    BASE_AGENT_PROMPT,
    CATEGORY_CLASSIFICATION_PROMPT,
    COMPLAINT_CLASSIFICATION_PROMPT,
    PRIORITY_INSTRUCTION_PROMPT,
    URGENCY_CLASSIFICATION_PROMPT,
    build_answer_quality_user_prompt,
    build_dynamic_category_prompt,
    build_quality_retry_feedback,
)
from support_agent.state import Category, SupportTicketState, Urgency
from support_agent.llm import build_chat_model
from langchain_core.tools.base import InjectedToolCallId

settings = Settings()


class ComplaintClassification(BaseModel):
    """Классифицирует, является ли обращение жалобой. В случае жалобы эскалирует на оператора."""
    is_complaint: bool

class UrgencyClassification(BaseModel):
    """Классифицирует срочность обращения. Может быть low, medium или high."""
    urgency: Urgency


class CategoryClassification(BaseModel):
    """Классифицирует категорию обращения. Может быть technical_question, billing_question, how_to или other."""
    category: Category


class AnswerQualityEvaluation(BaseModel):
    """Оценка качества ответа ассистента.
      quality_score - итоговая оценка 0..1, completeness - полнота 0..1, 
      politeness - вежливость 0..1, relevance - релевантность 0..1, 
      notes - короткие замечания на русском языке."""
    quality_score: float
    completeness: float
    politeness: float
    relevance: float
    notes: str


# Конструируем pipeline: промпт → модель со структурированным выводом
llm = build_chat_model(settings)
complaint_model = llm.with_structured_output(ComplaintClassification)
category_model = llm.with_structured_output(CategoryClassification)
urgency_model = llm.with_structured_output(UrgencyClassification)


complaint_classification = COMPLAINT_CLASSIFICATION_PROMPT | complaint_model
category_classification = CATEGORY_CLASSIFICATION_PROMPT | category_model
urgency_classification = URGENCY_CLASSIFICATION_PROMPT | urgency_model


def build_support_graph(
    settings: Settings,
    kb: HybridChromaKnowledgeBase,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:

    chat_model = build_chat_model(settings)
    dialog_agent = _build_dialog_agent(
        chat_model=chat_model,
        kb=kb,
        settings=settings,
    )

    graph = StateGraph(SupportTicketState)

    graph.add_node("ReceiveRequest", receive_request)
    graph.add_node("ClassifyComplaint", classify_complaint)
    graph.add_node("ClassifyUrgency", classify_urgency)
    graph.add_node("ClassifyCategory", classify_category)
    graph.add_node("GatherClassifications", gather_classifications)
    graph.add_node("EscalateComplaint", escalate_complaint_with_logging)
    graph.add_node("DialogAgent", dialog_agent)
    graph.add_node("FinalizeResponse", finalize_response)

    graph.set_entry_point("ReceiveRequest")
    graph.add_edge("ReceiveRequest", "ClassifyComplaint")
    graph.add_edge("ReceiveRequest", "ClassifyUrgency")
    graph.add_edge("ReceiveRequest", "ClassifyCategory")
    graph.add_edge(
        ["ClassifyComplaint", "ClassifyUrgency", "ClassifyCategory"],
        "GatherClassifications",
    )
    graph.add_conditional_edges(
        "GatherClassifications",
        route_after_gather,
        {
            "complaint": "EscalateComplaint",
            "non_complaint": "DialogAgent",
        },
    )

    graph.add_edge("DialogAgent", "FinalizeResponse")
    graph.add_edge("EscalateComplaint", END)
    graph.add_edge("FinalizeResponse", END)

    return graph.compile(checkpointer=checkpointer)


def _build_dialog_agent(chat_model: Any, kb: HybridChromaKnowledgeBase, settings: Settings) -> CompiledStateGraph:
    @tool("knowledge_base_search")
    def knowledge_base_search(query: str,
                            tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        """Search internal support knowledge base by user query.
        Args:
            query (str): The user's query.
        Returns:
            str: The search results.
        """
        try:
            docs = kb.search(query=query, filters=None, top_k=settings.kb_top_k)
        except Exception as exc:
            logger.exception("knowledge_base_search failed for query='{}'", query)
            node_exc = KnowledgeBaseSearchToolError(str(exc))
            raise ToolException(str(node_exc)) from node_exc

        parts = []
        for i, d in enumerate(docs):
            src = d.metadata.get("source") or d.metadata.get("filename") or "unknown"
            parts.append(f"\n[# Фрагмент {i} | source={src}]\n{d.page_content}")

        logger.info(f"Knowledge base search for query: '{query}' returned {len(parts)} relevant results.")

        return Command(update={
                "retrieved_docs": parts,
                "messages": [
            ToolMessage(content=f"База знаний вернула: {parts if parts else 'Документы не найдены'} ", tool_call_id=tool_call_id, name="knowledge_base_search")
        ],
    })
        

    @before_agent(state_schema=SupportTicketState)
    def apply_priority_policy(state: SupportTicketState, runtime: Runtime) -> dict[str, Any] | None:
        logger.info(
            "DialogAgent node started for ticket={}, category={}, urgency={}",
            state.get("ticket_id"),
            state.get("category"),
            state.get("urgency"),
        )
        urgent = state.get("urgency") in {"medium", "high"} 
        if not urgent:
            return None
        logger.info(f"Applying priority policy for ticket: {state.get('ticket_id')} with urgency: {state.get('urgency')}")
        return {
            "priority_instruction": PRIORITY_INSTRUCTION_PROMPT,
            "events": [
                make_event(
                    state,
                    "DialogAgent",
                    "priority_instruction_applied",
                    {
                        "urgency": state.get("urgency"),
                        "priority_instruction": state.get("priority_instruction"),
                    },
                )
            ],
        }

    @wrap_model_call(state_schema=SupportTicketState)
    def inject_dynamic_system_prompt(request: ModelRequest,
                                    handler: Callable[[ModelRequest], ModelResponse],
                                        ) -> ModelResponse:
        category = request.state.get("category", "other")
        model_retry_count = request.state.get("model_retry_count", 0)
        priority_instruction = request.state.get("priority_instruction")
        logger.info(f"Injecting dynamic system prompt for category: {category} with priority_instruction: {priority_instruction} for ticket: {request.state.get('ticket_id')}")
        dynamic_prompt = build_dynamic_category_prompt(category, priority_instruction)
        evaluation_notes = request.state.get("evaluation_notes")    

        if evaluation_notes and model_retry_count > 0:
            dynamic_prompt += f"\n\n[FEEDBACK_RETRY] Дополнительная информация для улучшения ответа: {evaluation_notes}"

        if request.system_message:
            request.system_message = SystemMessage(
                content=f"{request.system_message.content}\n\n{dynamic_prompt}"
            )
        else:
            request.system_message = SystemMessage(content=dynamic_prompt)
        return handler(request)

    @after_model(state_schema=SupportTicketState, can_jump_to=["model"])
    def retry_low_quality_answer(state: SupportTicketState, runtime: Runtime) -> dict[str, Any] | None:

        last_ai = _last_ai_message(state.get("messages", []))
        if last_ai is None or getattr(last_ai, "tool_calls", None):
            return None

        score, completeness, politeness, relevance, notes = _evaluate_answer_quality(_message_text(last_ai))
        logger.info(f"Evaluated answer quality: score={score}, completeness={completeness}, politeness={politeness}, relevance={relevance}, notes={notes} for ticket: {state.get('ticket_id')}"
        )
        retries = state.get("model_retry_count", 0)
        threshold = state.get("quality_threshold", settings.quality_threshold)
        max_retries = state.get("max_model_retries", settings.max_model_retries)
        if score >= threshold or retries >= max_retries:
            return {
                "completeness": completeness,
                "politeness": politeness,
                "relevance": relevance,
                "quality_score": score,
                "evaluation_notes": notes,
                "events": [
                    make_event(
                        state,
                        "DialogAgent",
                        "quality_ok",
                        {
                            "quality_score": score,
                            "threshold": threshold,
                            "retry_count": retries,
                        },
                    )
                ],
            }

        feedback = build_quality_retry_feedback(notes)
        return {
            "completeness": completeness,
            "politeness": politeness,
            "relevance": relevance,
            "quality_score": score,
            "evaluation_notes": feedback,
            "model_retry_count": retries + 1,
            "jump_to": "model",
            "events": [
                make_event(
                    state,
                    "DialogAgent",
                    "quality_retry_requested",
                    {
                        "quality_score": score,
                        "threshold": threshold,
                        "retry_count": retries + 1,
                        "notes": notes,
                    },
                )
            ],
        }

    return create_agent(
        model=chat_model,
        tools=[knowledge_base_search],
        system_prompt=BASE_AGENT_PROMPT,
        middleware=[apply_priority_policy, inject_dynamic_system_prompt, retry_low_quality_answer],
        state_schema=SupportTicketState,
        name="SupportDialogAgent",
    )


def receive_request(state: SupportTicketState) -> dict[str, Any]:
    logger.info("ReceiveRequest node started for ticket={}", state.get("ticket_id"))
    try:
        user_text = state.get("user_text") or _latest_user_query(state.get("messages", []))
        updates: dict[str, Any] = {
            "user_text": user_text,
            "quality_threshold": state.get("quality_threshold", 0.75),
            "max_model_retries": state.get("max_model_retries", 1),
            "model_retry_count": state.get("model_retry_count", 0),
            "events": [make_event(state, "ReceiveRequest", "request_received", {"ticket_id": state.get("ticket_id")})],
        }
        if not state.get("messages"):
            updates["messages"] = [HumanMessage(content=user_text)]
        return updates
    except Exception as exc:
        node_exc = ReceiveRequestNodeError(str(exc))
        logger.exception("ReceiveRequest node failed for ticket={}", state.get("ticket_id"))
        return {
            "has_error": True,
            "error_node": "ReceiveRequest",
            "errors": [make_error(state, "ReceiveRequest", type(node_exc).__name__, str(node_exc))],
            "events": [make_event(state, "ReceiveRequest", "request_failed", {"error": str(node_exc)})],
        }


def classify_complaint(state: SupportTicketState) -> dict[str, Any]:
    logger.info("ClassifyComplaint node started for ticket={}", state.get("ticket_id"))
    try:
        classification = cast(ComplaintClassification, complaint_classification.invoke({"text": state.get("user_text")}))
        logger.info(
            "ClassifyComplaint node completed for ticket={} with is_complaint={}",
            state.get("ticket_id"),
            classification.is_complaint,
        )
        return {
            "is_complaint": classification.is_complaint,
            "events": [
                make_event(
                    state,
                    "ClassifyComplaint",
                    "classification_done",
                    {"is_complaint": classification.is_complaint},
                )
            ],
        }
    except Exception as exc:
        node_exc = ClassifyComplaintNodeError(str(exc))
        logger.exception("ClassifyComplaint node failed for ticket={}", state.get("ticket_id"))
        return {
            "is_complaint": True,
            "errors": [make_error(state, "ClassifyComplaint", type(node_exc).__name__, str(node_exc))],
            "events": [make_event(state, "ClassifyComplaint", "classification_failed", {"error": str(node_exc)})],
        }


def classify_urgency(state: SupportTicketState) -> dict[str, Any]:
    logger.info("ClassifyUrgency node started for ticket={}", state.get("ticket_id"))
    try:
        classification = cast(UrgencyClassification, urgency_classification.invoke({"text": state.get("user_text")}))
        logger.info(
            "ClassifyUrgency node completed for ticket={} with urgency={}",
            state.get("ticket_id"),
            classification.urgency,
        )
        return {
            "urgency": classification.urgency,
            "events": [make_event(state, "ClassifyUrgency", "classification_done", {"urgency": classification.urgency})],
        }
    except Exception as exc:
        node_exc = ClassifyUrgencyNodeError(str(exc))
        logger.exception("ClassifyUrgency node failed for ticket={}", state.get("ticket_id"))
        return {
            "urgency": "medium",
            "errors": [make_error(state, "ClassifyUrgency", type(node_exc).__name__, str(node_exc))],
            "events": [make_event(state, "ClassifyUrgency", "classification_failed", {"error": str(node_exc)})],
        }


def classify_category(state: SupportTicketState) -> dict[str, Any]:
    logger.info("ClassifyCategory node started for ticket={}", state.get("ticket_id"))
    try:
        classification = cast(CategoryClassification, category_classification.invoke({"text": state.get("user_text")}))
        logger.info(
            "ClassifyCategory node completed for ticket={} with category={}",
            state.get("ticket_id"),
            classification.category,
        )
        return {
            "category": classification.category,
            "events": [make_event(state, "ClassifyCategory", "classification_done", {"category": classification.category})],
        }
    except Exception as exc:
        node_exc = ClassifyCategoryNodeError(str(exc))
        logger.exception("ClassifyCategory node failed for ticket={}", state.get("ticket_id"))
        return {
            "category": "other",
            "errors": [make_error(state, "ClassifyCategory", type(node_exc).__name__, str(node_exc))],
            "events": [make_event(state, "ClassifyCategory", "classification_failed", {"error": str(node_exc)})],
        }





def gather_classifications(state: SupportTicketState) -> dict[str, Any]:
    logger.info(
        "GatherClassifications node started for ticket={} (is_complaint={}, category={}, urgency={})",
        state.get("ticket_id"),
        state.get("is_complaint"),
        state.get("category"),
        state.get("urgency"),
    )
    try:
        return {
            "is_complaint": state.get("is_complaint"),
            "category": state.get("category"),
            "urgency": state.get("urgency"),
            "events": [
                make_event(
                    state,
                    "GatherClassifications",
                    "gather_completed",
                    {
                        "is_complaint": state.get("is_complaint"),
                        "category": state.get("category"),
                        "urgency": state.get("urgency"),
                    },
                )
            ],
        }
    except Exception as exc:
        node_exc = GatherClassificationsNodeError(str(exc))
        logger.exception("GatherClassifications node failed for ticket={}", state.get("ticket_id"))
        return {
            "has_error": True,
            "error_node": "GatherClassifications",
            "errors": [make_error(state, "GatherClassifications", type(node_exc).__name__, str(node_exc))],
            "events": [make_event(state, "GatherClassifications", "gather_failed", {"error": str(node_exc)})],
        }


def escalate_complaint_with_logging(state: SupportTicketState) -> dict[str, Any]:
    logger.info(
        "EscalateComplaint node started for ticket={} (category={}, urgency={})",
        state.get("ticket_id"),
        state.get("category"),
        state.get("urgency"),
    )
    try:
        result = escalate_complaint(state)
        logger.info("EscalateComplaint node completed for ticket={}", state.get("ticket_id"))
        return result
    except Exception as exc:
        node_exc = EscalateComplaintNodeError(str(exc))
        logger.exception("EscalateComplaint node failed for ticket={}", state.get("ticket_id"))
        return {
            "has_error": True,
            "error_node": "EscalateComplaint",
            "errors": [make_error(state, "EscalateComplaint", type(node_exc).__name__, str(node_exc))],
            "events": [make_event(state, "EscalateComplaint", "escalation_failed", {"error": str(node_exc)})],
        }


def finalize_response(state: SupportTicketState) -> dict[str, Any]:
    logger.info("FinalizeResponse node started for ticket={}", state.get("ticket_id"))
    try:
        final_text = _last_final_ai_text(state.get("messages", []))
        if not final_text and state.get("category") == "other":
            final_text = "По этой теме я не поддерживаю диалог."
        logger.info(
            "FinalizeResponse node completed for ticket={} (response_len={})",
            state.get("ticket_id"),
            len(final_text),
        )
        return {
            "final_response": final_text,
            "escalated": False,
            "escalation_reason": None,
        }
    except Exception as exc:
        node_exc = FinalizeResponseNodeError(str(exc))
        logger.exception("FinalizeResponse node failed for ticket={}", state.get("ticket_id"))
        return {
            "has_error": True,
            "error_node": "FinalizeResponse",
            "errors": [make_error(state, "FinalizeResponse", type(node_exc).__name__, str(node_exc))],
            "events": [make_event(state, "FinalizeResponse", "finalization_failed", {"error": str(node_exc)})],
        }


def route_after_gather(state: SupportTicketState) -> Literal["complaint", "non_complaint"]:
    return "complaint" if bool(state.get("is_complaint")) else "non_complaint"


def _latest_user_query(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_text(message)
    return ""


def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _last_final_ai_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        if getattr(message, "tool_calls", None):
            continue
        text = _message_text(message).strip()
        if not text:
            continue
        lower = text.lower()
        if "retrieving function output" in lower:
            continue
        return text
    return ""


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _evaluate_answer_quality(answer: str) -> tuple[float, float, float, float, str]:
    normalized = answer.strip()
    if not normalized:
        return 0.0, 0.0, 0.0, 0.0, "пустой или нерелевантный ответ"

    try:
        evaluator = llm.with_structured_output(AnswerQualityEvaluation)
        result = cast(
            AnswerQualityEvaluation,
            evaluator.invoke(
                [
                    SystemMessage(
                        content=ANSWER_QUALITY_EVALUATION_SYSTEM_PROMPT
                    ),
                    HumanMessage(content=build_answer_quality_user_prompt(normalized)),
                ]
            ),
        )
        return (
            round(max(0.0, min(1.0, result.quality_score)), 3),
            round(max(0.0, min(1.0, result.completeness)), 3),
            round(max(0.0, min(1.0, result.politeness)), 3),
            round(max(0.0, min(1.0, result.relevance)), 3),
            result.notes.strip() or "замечаний нет",
        )
    except Exception:
        # объяснение: если модель не может корректно оценить, используем простые эвристики для базовой оценки качества ответа
        answer_lower = normalized.lower()
        completeness = min(1.0, max(0.1, len(normalized) / 220.0))
        politeness = 1.0 if any(token in answer_lower for token in ("спасибо", "пожалуйста", "извин")) else 0.7
        relevance = 0.95 if normalized else 0.1
        notes_parts: list[str] = []
        if len(normalized) < 40:
            notes_parts.append("ответ слишком короткий")
        if politeness < 0.9:
            notes_parts.append("не хватает вежливой формулировки")
        if relevance < 0.5:
            notes_parts.append("пустой или нерелевантный ответ")
        notes = "; ".join(notes_parts) if notes_parts else "замечаний нет"
        quality_score = round(max(0.0, min(1.0, 0.4 * completeness + 0.2 * politeness + 0.4 * relevance)), 3)
        return quality_score, round(completeness, 3), round(politeness, 3), round(relevance, 3), notes
