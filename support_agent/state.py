from __future__ import annotations

from typing import Any, Literal, TypedDict


Category = Literal["complaint", "technical_question", "billing_question", "how_to", "other"]
Sentiment = Literal["negative", "neutral", "positive"]
Urgency = Literal["low", "medium", "high"]


class SupportTicketState(TypedDict, total=False):
    ticket_id: str
    user_text: str

    category: Category
    is_complaint: bool
    urgency: Urgency
    sentiment: Sentiment

    search_query: str
    search_filters: dict[str, Any]
    extracted_entities: dict[str, Any]
    retrieved_docs: list[dict[str, Any]]

    draft_response: str
    final_response: str

    completeness: float
    politeness: float
    relevance: float
    quality_score: float
    evaluation_notes: str
    quality_threshold: float

    refinement_count: int
    max_refinements: int

    escalated: bool
    escalation_reason: str | None

    current_node: str
    has_error: bool
    error_node: str
    errors: list[dict[str, Any]]
    events: list[dict[str, Any]]
