from __future__ import annotations

import json

from support_agent.state import SupportTicketState





def build_classification_prompt(state: SupportTicketState) -> tuple[str, str]:
    system_prompt = (
        "Ты классифицируешь обращения в службу поддержки. Главная задача - понять,"
        "является ли обращение жалобой, к жалобам также отнеси бессмысленные негативные отзывы."
        " Второстепенная задача - определить категорию обращения. "
        "Верни структурированный ответ с полями: is_complaint (bool), category."
    )
    user_prompt = (
        f"Ticket text:\n{state.get('user_text')}\n\n"
        "Allowed category values: complaint, technical_question, billing_question, how_to, other.\n"
    )
    return system_prompt, user_prompt


def build_generation_prompt(state: SupportTicketState) -> tuple[str, str]:
    docs = state.get("retrieved_docs", [])
    docs_block = "\n\n".join(
        f"[{doc.get('id')}] {doc.get('title')}\n{doc.get('content')}" for doc in docs
    )

    system_prompt = (
        "You are a support assistant. Be polite, concise, and factual. "
        "Use only provided context. If context is insufficient, explicitly say so."
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
        "You improve support responses based on review feedback. "
        "Keep answer polite, complete, and relevant."
    )
    user_prompt = (
        f"Ticket text:\n{state.get('user_text', '')}\n\n"
        f"Current answer:\n{state.get('draft_response', '')}\n\n"
        f"Reviewer notes:\n{state.get('evaluation_notes', '')}\n\n"
        "Rewrite the answer to address the notes."
    )
    return system_prompt, user_prompt
