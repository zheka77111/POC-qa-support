from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, TypedDict


Category = Literal["complaint", "technical_question", "billing_question", "how_to", "other"]


class SupportTicketState(TypedDict, total=False):
    ticket_id: str
    user_text: str

    category: Category
    is_complaint: bool

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
    errors: Annotated[list[dict[str, Any]], add]
    events: Annotated[list[dict[str, Any]], add]
