from __future__ import annotations

import logging
from functools import partial
from typing import Annotated, Any, Callable, Literal, cast
from loguru import logger
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, after_model, before_agent, wrap_model_call
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from langgraph.types import Command
from support_agent.config import Settings
from support_agent.knowledge_base import HybridChromaKnowledgeBase
from support_agent.logging_utils import make_error, make_event
from support_agent.nodes.escalation import escalate_complaint
from support_agent.state import Category, SupportTicketState, Urgency
from support_agent.llm import build_chat_model
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.prompts import ChatPromptTemplate
from langgraph.runtime import Runtime

BASE_AGENT_PROMPT = (
    "Ты диалоговый помощник техподдержки. "
    "Веди диалог только по базе знаний и отвечай коротко, вежливо и по делу. "
    "Для категорий technical_question, billing_question, how_to сначала используй инструмент knowledge_base_search. "
    "Если инструмент вернул NOT_FOUND, ответь ровно: 'Не знаю'. "
    "Если категория обращения == other, ответь ровно: 'По этой теме я не поддерживаю диалог.' и не вызывай инструменты."
)

settings = Settings()


class ComplaintClassification(BaseModel):
    """Классифицирует, является ли обращение жалобой. В случае жалобы эскалирует на оператора."""
    is_complaint: bool

class RelevantChunkClassification(BaseModel):
    """Классифицирует релевантность найденных в базе знаний фрагментов. 
    Вернет список bool для каждого фрагмента, где True - релевантный, False - нерелевантный."""
    relevant_chunks: list[bool]

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


BASE_AGENT_PROMPT = (
    "Ты диалоговый помощник техподдержки. "
    "Веди диалог только по базе знаний и отвечай коротко, вежливо и по делу. "
    "Для категорий technical_question, billing_question, how_to сначала используй инструмент knowledge_base_search. "
    "Если инструмент вернул NOT_FOUND, ответь ровно: 'Не знаю'. "
    "Если категория обращения == other, ответь ровно: 'По этой теме я не поддерживаю диалог.' и не вызывай инструменты."
)


urgency_classification_prompt = ChatPromptTemplate.from_messages([(
    "system",
    "Ты классифицируешь срочность обращений в службу поддержки. "
    "Верни структурированный ответ с полем: urgency (str)."
),("human",
   "{text}")])

category_classification_prompt = ChatPromptTemplate.from_messages([
    (
    "system",
    "Ты классифицируешь категории обращений в службу поддержки. "
    "Верни структурированный ответ с полем: category."
), 
("human", "{text}")
])

classify_complaint_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Ты — эксперт по классификации обращений клиентов.
        
ЗАДАЧА: Определи, является ли обращение жалобой.

КРИТЕРИИ ЖАЛОБЫ (is_complaint=True):
- Клиент выражает недовольство качеством товара/услуги
- Есть требование компенсации, возврата денег, извинений
- Упоминание негативного опыта, расстройства, гнева
- Угроза прекратить сотрудничество
- Жалоба на персонал, сроки, брак, недостаточный сервис

НЕ ЖАЛОБА (is_complaint=False):
- Обычный вопрос о сервисе, о процессах
- Есть проблема с использованием сервисов / услуг / функций
- Запрос информации, консультации
- Положительный отзыв, благодарность
- Предложение улучшить сервис (без недовольства)
- Техническая проблема без эмоциональной окраски

ВАЖНО: Будь строгим критериям. Если есть хоть малейшее недовольство — помечай как жалобу."""
    ),
    ("human", "{text}")
])


# Конструируем pipeline: промпт → модель со структурированным выводом
llm = build_chat_model(settings)
complaint_model = llm.with_structured_output(ComplaintClassification)
category_model = llm.with_structured_output(CategoryClassification)
urgency_model = llm.with_structured_output(UrgencyClassification)


complaint_classification = classify_complaint_prompt | complaint_model
category_classification = category_classification_prompt | category_model
urgency_classification = urgency_classification_prompt | urgency_model


def build_support_graph(
    settings: Settings,
    logger: logger,
    kb: HybridChromaKnowledgeBase,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    _ = logger
    chat_model = _build_chat_model(settings)
    dialog_agent = _build_dialog_agent(
        chat_model=chat_model,
        kb=kb,
        settings=settings,
    )

    graph = StateGraph(SupportTicketState)

    graph.add_node("ReceiveRequest", receive_request)
    graph.add_node("ClassifyComplaint", partial(classify_complaint, chat_model=chat_model))
    graph.add_node("ClassifyUrgency", partial(classify_urgency, chat_model=chat_model))
    graph.add_node("ClassifyCategory", partial(classify_category, chat_model=chat_model))
    graph.add_node("GatherClassifications", gather_classifications)
    graph.add_node("EscalateComplaint", escalate_complaint)
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
    # graph.add_conditional_edges(
    #     "GatherClassifications",
    #     route_after_gather,
    #     {
    #         "complaint": "EscalateComplaint",
    #         "non_complaint": "DialogAgent",
    #     },
    # )

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
        docs = kb.search(query=query, filters=None, top_k=settings.kb_top_k)

        parts = []
        for i, d in enumerate(docs):
            src = d.metadata.get("source") or d.metadata.get("filename") or "unknown"
            parts.append(f"\n[# Фрагмент {i} | source={src}]\n{d.page_content}")

        relevant_parts: list[str] = []
        if parts:
            llm_structured = llm.with_structured_output(RelevantChunkClassification)
            response = llm_structured.invoke([
                SystemMessage(content=f"""Ты — эксперт по оценке релевантности ответов поддержки.
        ЗАДАЧА: Оцени релевантность данных, извлеченных из базы знаний по отношению к вопросу клиента.
        На вход поступают несколько фрагментов из базы знаний и вопрос клиента.
        Твоя задача — оценить, насколько эти фрагменты релевантны для ответа на вопрос клиента.
        Вот фрагменты из базы знаний: {parts}. Вот запрос клиента:"""),
                HumanMessage(content=f"{query}"),
            ])

            response = cast(RelevantChunkClassification, response)
            relevant_parts = [part for part, is_relevant in zip(parts, response.relevant_chunks) if is_relevant]

        if relevant_parts:
            result: str | list[str] = relevant_parts
        else:
            result = "Документы найдены, но не релевантные для ответа на вопрос клиента."
        logger.info(f"Knowledge base search for query: '{query}' returned {len(relevant_parts)} relevant results.")

        return Command(update={
                "retrieved_docs": result if result else [],
                "messages": [
            ToolMessage(content=f"База знаний вернула: {result if result else 'NOT_FOUND'} ", tool_call_id=tool_call_id, name="knowledge_base_search")
        ],
    })
        

    @before_agent(state_schema=SupportTicketState)
    def apply_priority_policy(state: SupportTicketState, runtime: Any) -> dict[str, Any] | None:
        _ = runtime
        urgent = state.get("urgency") in {"medium", "high"} or bool(state.get("is_return_in_3_days"))
        if not urgent:
            return None
        return {
            "priority_instruction": "Пользователю необходимо решить вопрос максимально срочно!",
            "events": [
                make_event(
                    state,
                    "DialogAgent",
                    "priority_instruction_applied",
                    {
                        "urgency": state.get("urgency"),
                        "is_return_in_3_days": state.get("is_return_in_3_days"),
                    },
                )
            ],
        }

    @wrap_model_call(state_schema=SupportTicketState)
    def inject_dynamic_system_prompt(request: ModelRequest,
                                    handler: Callable[[ModelRequest], ModelResponse],
                                        ) -> ModelResponse:
        category = request.state.get("category", "other")
        dynamic_prompt_parts = [f"Категория обращения: {category}."]
        priority_instruction = request.state.get("priority_instruction")
        if priority_instruction:
            dynamic_prompt_parts.append(priority_instruction)
        dynamic_prompt = "\n".join(dynamic_prompt_parts)

        if request.system_message:
            request.system_message = SystemMessage(
                content=f"{request.system_message.content}\n\n{dynamic_prompt}"
            )
        else:
            request.system_message = SystemMessage(content=dynamic_prompt)
        return handler(request)

    @after_model(state_schema=SupportTicketState, can_jump_to=["model"])
    def retry_low_quality_answer(state: SupportTicketState, runtime: Any) -> dict[str, Any] | None:
        _ = runtime
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

        feedback = (
            "Качество ответа ниже порога. "
            f"Проблемы: {notes}. "
            "Перегенерируй ответ: исправь полноту, вежливость и релевантность; "
            "используй knowledge_base_search, если нужны факты; "
            "если данных нет, ответь ровно 'Не знаю'."
        )
        return {
            "completeness": completeness,
            "politeness": politeness,
            "relevance": relevance,
            "quality_score": score,
            "evaluation_notes": feedback,
            "model_retry_count": retries + 1,
            "messages": [HumanMessage(content=feedback)],
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


def classify_complaint(state: SupportTicketState, chat_model: Any) -> dict[str, Any]:
    try:
        classification = cast(ComplaintClassification, complaint_classification.invoke({"text": state.get("user_text")}))
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
        return {
            "is_complaint": True,
            "errors": [make_error(state, "ClassifyComplaint", type(exc).__name__, str(exc))],
            "events": [make_event(state, "ClassifyComplaint", "classification_failed", {"error": str(exc)})],
        }


def classify_urgency(state: SupportTicketState, chat_model: Any) -> dict[str, Any]:
    try:
        classification = cast(UrgencyClassification, urgency_classification.invoke({"text": state.get("user_text")}))
        return {
            "urgency": classification.urgency,
            "events": [make_event(state, "ClassifyUrgency", "classification_done", {"urgency": classification.urgency})],
        }
    except Exception as exc:
        return {
            "urgency": "medium",
            "errors": [make_error(state, "ClassifyUrgency", type(exc).__name__, str(exc))],
            "events": [make_event(state, "ClassifyUrgency", "classification_failed", {"error": str(exc)})],
        }


def classify_category(state: SupportTicketState, chat_model: Any) -> dict[str, Any]:
    try:
        classification = cast(CategoryClassification, category_classification.invoke({"text": state.get("user_text")}))
        return {
            "category": classification.category,
            "events": [make_event(state, "ClassifyCategory", "classification_done", {"category": classification.category})],
        }
    except Exception as exc:
        return {
            "category": "other",
            "errors": [make_error(state, "ClassifyCategory", type(exc).__name__, str(exc))],
            "events": [make_event(state, "ClassifyCategory", "classification_failed", {"error": str(exc)})],
        }





def gather_classifications(state: SupportTicketState) -> dict[str, Any]:
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


def finalize_response(state: SupportTicketState) -> dict[str, Any]:
    last_ai = _last_ai_message(state.get("messages", []))
    final_text = _message_text(last_ai) if last_ai else ""
    return {
        "messages":[AIMessage(content=final_text)],
        "final_response": final_text,
        "escalated": False,
        "escalation_reason": None,
        "events": [make_event(state, "FinalizeResponse", "completed")],
    }


def route_after_gather(state: SupportTicketState) -> Literal["complaint", "non_complaint"]:
    return "complaint" if bool(state.get("is_complaint")) else "non_complaint"


def _build_chat_model(settings: Settings) -> Any:
    if settings.llm_provider != "gigachat":
        raise RuntimeError("Dialog mode currently supports llm_provider='gigachat' only.")
    if not settings.gigachat_credentials:
        raise RuntimeError("GIGACHAT_API_KEY is required for gigachat provider.")

    from langchain_gigachat import GigaChat

    return GigaChat(
        credentials=settings.gigachat_credentials,
        scope=settings.gigachat_scope,
        model=settings.gigachat_model,
        verify_ssl_certs=False,
        timeout=settings.timeout_seconds,
        top_p=settings.top_p,
    )


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


def _structured_value(output: Any, key: str, default: Any) -> Any:
    if isinstance(output, dict):
        return output.get(key, default)
    return getattr(output, key, default)


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
                        content=(
                            "Ты оцениваешь качество ответа ассистента техподдержки.\n"
                            "Верни строго структурированный результат:\n"
                            "- quality_score: итоговая оценка 0..1\n"
                            "- completeness: полнота 0..1\n"
                            "- politeness: вежливость 0..1\n"
                            "- relevance: релевантность 0..1\n"
                            "- notes: короткие замечания на русском языке\n"
                            "Оцени строго по шкале 0..1."
                        )
                    ),
                    HumanMessage(content=f"Оцени этот ответ:\n{normalized}"),
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
