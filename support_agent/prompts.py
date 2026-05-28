from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from support_agent.state import SupportTicketState


BASE_AGENT_PROMPT = (
    "Ты диалоговый помощник техподдержки. "
    "Веди диалог только по базе знаний и отвечай коротко, вежливо и по делу. "
    "Для категорий technical_question, billing_question, how_to сначала используй инструмент knowledge_base_search. "
    "Если инструмент вернул NOT_FOUND, ответь ровно: 'Не знаю'. "
    "Если категория обращения == other, ответь ровно: 'По этой теме я не поддерживаю диалог.' и не вызывай инструменты."
)


URGENCY_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты классифицируешь срочность обращений в службу поддержки. "
            "Верни структурированный ответ с полем: urgency (str).",
        ),
        ("human", "{text}"),
    ]
)


CATEGORY_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты классифицируешь категории обращений в службу поддержки. "
            "Верни структурированный ответ с полем: category.",
        ),
        ("human", "{text}"),
    ]
)


COMPLAINT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
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

ВАЖНО: Будь строгим критериям. Если есть хоть малейшее недовольство — помечай как жалобу.""",
        ),
        ("human", "{text}"),
    ]
)


PRIORITY_INSTRUCTION_PROMPT = "Пользователю необходимо решить вопрос максимально срочно!"


ANSWER_QUALITY_EVALUATION_SYSTEM_PROMPT = (
    "Ты оцениваешь качество ответа ассистента техподдержки.\n"
    "Верни строго структурированный результат:\n"
    "- quality_score: итоговая оценка 0..1\n"
    "- completeness: полнота 0..1\n"
    "- politeness: вежливость 0..1\n"
    "- relevance: релевантность 0..1\n"
    "- notes: короткие замечания на русском языке\n"
    "Оцени строго по шкале 0..1."
)


def build_dynamic_category_prompt(category: str, priority_instruction: str | None = None) -> str:
    parts = [f"Категория обращения: {category}."]
    if priority_instruction:
        parts.append(priority_instruction)
    return "\n".join(parts)


def build_quality_retry_feedback(notes: str) -> str:
    return (
        "Качество ответа ниже порога. "
        f"Проблемы: {notes}. "
        "Перегенерируй ответ: исправь полноту, вежливость и релевантность; "
        "используй knowledge_base_search, если нужны факты; "
        "если данных нет, ответь ровно 'Не знаю'."
    )


def build_answer_quality_user_prompt(answer: str) -> str:
    return f"Оцени этот ответ:\n{answer}"





def build_classification_prompt(state: SupportTicketState) -> tuple[str, str]:
    system_prompt = (
        "Ты классифицируешь обращения в службу поддержки. Главная задача - понять,"
        "является ли обращение жалобой, к жалобам также отнеси бессмысленные негативные отзывы."
        " Второстепенная задача - определить категорию обращения. "
        "Верни структурированный ответ с полями: is_complaint (bool), category."
    )
    user_prompt = (
        f"Ticket text:\n{state.get('user_text')}\n\n"
        "Allowed category values: technical_question, billing_question, how_to, other.\n"
    )
    return system_prompt, user_prompt


def build_generation_prompt(state: SupportTicketState) -> tuple[str, str]:
    docs = state.get("retrieved_docs", [])
    docs_parts: list[str] = []
    for doc in docs:
        if hasattr(doc, "page_content"):
            metadata = getattr(doc, "metadata", {}) or {}
            docs_parts.append(f"[ticket_{metadata.get('id', 'unknown')}] содержит {doc.page_content}")
        else:
            docs_parts.append(str(doc))
    docs_block = "\n\n".join(docs_parts)

    system_prompt = (
        "ты агент поддержки. Будь вежлив, краток и точен. "
        "Используй только предоставленный контекст. Если контекста недостаточно, явно скажи об этом."
    )
    dynamic_payload = {
        "category": state.get("category"),
        "search_filters": state.get("search_filters", {}),
        "extracted_entities": state.get("extracted_entities", {}),
    }
    user_prompt = (
        f"Ticket text:\n{state.get('user_text', '')}\n\n"
        f"Dynamic context:\n{json.dumps(dynamic_payload, ensure_ascii=False)}\n\n"
        f"Knowledge base context:\n{docs_block}\n\n"
        "Write the best possible reply for the user."
    )
    return system_prompt, user_prompt


def build_evaluation_prompt(state: SupportTicketState) -> tuple[str, str]:
    system_prompt = (
        "Ты оцениваешь качество ответа поддержки. Верни JSON с ключами: "
        "completeness, politeness, relevance, notes. "
        "Каждый балл должен быть числом в диапазоне [0,1]."
    )
    user_prompt = (
        f"Ticket text:\n{state.get('user_text', '')}\n\n"
        f"Answer draft:\n{state.get('draft_response', '')}\n\n"
        "Evaluate by completeness, politeness, relevance."
    )
    return system_prompt, user_prompt


def build_refinement_prompt(state: SupportTicketState) -> tuple[str, str]:
    system_prompt = (
        "Ты улучшаешь ответы поддержки на основе обратной связи. "
        "Сделай ответ вежливым, полным и релевантным."
    )
    user_prompt = (
        f"Ticket text:\n{state.get('user_text', '')}\n\n"
        f"Current answer:\n{state.get('draft_response', '')}\n\n"
        f"Reviewer notes:\n{state.get('evaluation_notes', '')}\n\n"
        "Rewrite the answer to address the notes."
    )
    return system_prompt, user_prompt
