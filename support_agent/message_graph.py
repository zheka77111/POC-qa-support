


from functools import partial
from typing import Annotated, Any, Literal, cast
from loguru._logger import Logger as LoguruLogger
from langchain.agents import create_agent
from langchain.agents.middleware import after_model, before_agent, wrap_model_call
from langchain.chat_models import BaseChatModel
from langchain.tools import InjectedState, ToolRuntime
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.documents import Document
from langchain_core.tools import ToolException, tool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from langgraph.types import Command, Send
from support_agent.config import Settings
from support_agent.knowledge_base import HybridChromaKnowledgeBase
from support_agent.prompts import build_generation_prompt
from support_agent.state import Category, SupportTicketState, Urgency
from support_agent.state import SupportTicketState
from support_agent.logging_utils import make_event, setup_logger, append_event, append_error
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from langgraph.runtime import Runtime


from support_agent.utils import _latest_user_query
from support_agent.llm import build_chat_model

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


BASE_AGENT_PROMPT = (
    "Ты диалоговый помощник техподдержки. "
    "Веди диалог только по базе знаний и отвечай коротко, вежливо и по делу. "
    "Для категорий technical_question, billing_question, how_to сначала используй инструмент knowledge_base_search. "
    "Если инструмент вернул NOT_FOUND, ответь ровно: 'Не знаю'. "
    "Если категория обращения == other, ответь ровно: 'По этой теме я не поддерживаю диалог.' и не вызывай инструменты."
)

relevant_chunk_classification_prompt = (
    "Ты классифицируешь обращения в службу поддержки. Главная задача - понять,"
    "является ли обращение жалобой, к жалобам также отнеси бессмысленные негативные отзывы."

)


urgency_classification_prompt = (
    "Ты классифицируешь срочность обращений в службу поддержки. "
    "Верни структурированный ответ с полем: urgency (str)."
)

category_classification_prompt = ChatPromptTemplate.from_messages([(
    "system",
    "Ты классифицируешь категории обращений в службу поддержки. "
    "Верни структурированный ответ с полем: category."
), ("human", "{text}")
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
complaint_classification = classify_complaint_prompt | complaint_model


urgency_model = llm.with_structured_output(UrgencyClassification)
category_model = llm.with_structured_output(CategoryClassification)
category_classification = category_classification_prompt | category_model








def build_support_graph(
    settings: Settings,
    logger: LoguruLogger,
    checkpointer: InMemorySaver,
    kb: HybridChromaKnowledgeBase,
) -> CompiledStateGraph:
    
    def complaint_classification_node(state: SupportTicketState) -> dict[str, Any]:
        user_text = state.get("user_text") 
        classification = cast(ComplaintClassification, complaint_classification.invoke({"text": user_text}))
        logger.info(f"Complaint classification result: {classification.is_complaint} for ticket: {state.get('ticket_id')}")
        return {"is_complaint": classification.is_complaint}
    
    def category_classification_node(state: SupportTicketState) -> dict[str, Any]:
        user_text = state.get("user_text") 
        classification = cast(CategoryClassification, category_classification.invoke({"text": user_text}))
        logger.info(f"Category classification result: {classification.category} for ticket: {state.get('ticket_id')}")
        return {"category": classification.category}

    def knowledge_base_search(state: SupportTicketState, runtime: Runtime[Settings])-> dict[str, Any]:
        """
        Используетсы для ответов на вопрос пользователя, ищет данные в базе знаний.
        Используется, когда нужно найти информацию в документах.
        Args: 
        query: str - поисковый запрос
        top_k: int - количество результатов для возврата из базы знаний, по умолчанию 3
        Result: str - релевантный текст для ответа
        """
        query = state.get("user_text", "")
        docs:list[Document] = kb.search(query, top_k=runtime.context.kb_top_k if runtime.context.kb_top_k is not None else 3)
        if not docs:
            result = "Документов в базе знаний не найдено"

        parts = []
        for i, d in enumerate(docs):
            src = d.metadata.get("source") or d.metadata.get("filename") or "unknown"
            parts.append(f"\n[# Фрагмент {i} | source={src}]\n{d.page_content}")
        

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
            result = relevant_parts
        else:
            result = "Документы найдены, но не релевантные для ответа на вопрос клиента."
        logger.info(f"Knowledge base search for query: '{query}' returned {len(relevant_parts)} relevant results.")
        
        return {"retrieved_docs": result}
    
    def model_response_node(state: SupportTicketState) -> dict[str, Any]:
        retrieved_docs = state.get("retrieved_docs", [])
        if isinstance(retrieved_docs, list):
            response = "\n".join([doc.page_content for doc in retrieved_docs])
        else:
            response = retrieved_docs
        messages = state["messages"]
        system_prompt, user_prompt = build_generation_prompt(state)
        full_messages = [SystemMessage(content=system_prompt)] + list(messages)
        response = llm.invoke(user_prompt)
        return {"draft_response": response}
            

    
    

    workflow = StateGraph(SupportTicketState)
    workflow.add_node("complaint_classification", complaint_classification_node)
    workflow.add_node("category_classification", category_classification_node)
    workflow.add_node("knowledge_base_search", knowledge_base_search)
    workflow.add_node("model_response", model_response_node)

    

