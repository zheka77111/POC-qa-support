from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal
from langchain_core.documents import Document
from langchain.agents import AgentState

Category = Literal["technical_question", "billing_question", "how_to", "other"]
Urgency = Literal["low", "medium", "high"]


class SupportTicketState(AgentState):
    ticket_id: str
    user_text: str

    category: Category
    is_complaint: bool
    urgency: Urgency

    search_query: str
    retrieved_docs: list[Document]

    draft_response: str
    final_response: str

    completeness: float
    politeness: float
    relevance: float
    quality_score: float
    evaluation_notes: str
    quality_threshold: float
    priority_instruction: str

    model_retry_count: int
    max_model_retries: int

    escalated: bool
    escalation_reason: str | None
    refinement_count: int

    has_error: bool
    error_node: str
    errors: Annotated[list[dict[str, Any]], add]
    events: Annotated[list[dict[str, Any]], add]
