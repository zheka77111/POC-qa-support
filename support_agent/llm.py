from __future__ import annotations

import json
import re
from typing import Any, Mapping, TypeVar, overload

from pydantic import BaseModel, Field

from support_agent.config import Settings
from support_agent.state import Category


StructuredOutputModel = TypeVar("StructuredOutputModel", bound=BaseModel)
StructuredOutputSchema = Mapping[str, Any] | type[BaseModel]


class LLMClient:
    def invoke_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise NotImplementedError

    @overload
    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[StructuredOutputModel],
    ) -> StructuredOutputModel: ...

    @overload
    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: StructuredOutputSchema,
    ) -> BaseModel | dict[str, Any]:
        raise NotImplementedError

    def invoke_text(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class TicketClassification(BaseModel):
    """Классификация тикета поддержки для определения дальнейшей логики обработки."""

    is_complaint: bool = Field(
        description="Является ли тикет жалобой, которая должна быть эскалирована."
    )
    category: Category = Field(
        description="Категория тикета поддержки, например, жалоба, технический вопрос, вопрос о биллинге, как сделать или другое."
    )


class AnswerEvaluation(BaseModel):
    """Оценка качества ответа поддержки по нескольким осям для принятия решения об эскалации и улучшения ответа."""

    completeness: float = Field(
        ge=0,
        le=1,
        description="Насколько полно ответ, от 0 до 1.",
    )
    politeness: float = Field(
        ge=0,
        le=1,
        description="Насколько вежливо составлен ответ, от 0 до 1.",
    )
    relevance: float = Field(
        ge=0,
        le=1,
        description="Насколько релевантен ответ, от 0 до 1.",
    )
    notes: str = Field(description="Short review notes for improving the answer.")



class MockLLMClient(LLMClient):
    def invoke_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        text = user_prompt.lower()
        ticket_text = _extract_ticket_text(user_prompt).lower()

        if "classify support tickets" in system_prompt.lower() or "is_complaint" in system_prompt:
            is_complaint = any(
                marker in ticket_text
                for marker in [
                    "возмущ",
                    "неприемлемо",
                    "жалоб",
                    "издевательство",
                    "поддержка молчит",
                    "недопустимо",
                    "устал писать",
                    "не помогает",
                    "почему мой тикет закрыли",
                    "никто не связался",
                    "не реагирует",
                    "не работает сервис",
                    "пропали данные",
                    "списали деньги дважды",
                    "за что я плачу",
                    "четвертый раз объясняю",
                    "поддержка молчит",
                    "обещали решить",
                    "ничего не изменилось",
                    "тикет закрыли",
                    "никто не помогает",
                    "оператор недоступен",
                    "сроки решения",
                    "пропали данные",
                    "поддержка отвечает шаблонами",
                    "тормозит уже месяц",
                    "никакой реакции",
                    "статус не меняется",
                    "вылетает сразу",
                    "никто так и не связался",
                    "услуга не активировалась",
                    "условия стали хуже",
                    "система неверно считает баланс",
                ]
            )

            if is_complaint:
                category = "complaint"
            elif any(k in ticket_text for k in ["оплат", "карт", "billing"]):
                category = "billing_question"
            elif any(k in ticket_text for k in ["ошибка", "500", "не работает"]):
                category = "technical_question"
            elif any(k in ticket_text for k in ["как", "где", "можно ли"]):
                category = "how_to"
            else:
                category = "other"

            return {
                "is_complaint": is_complaint,
                "category": category,
            }

        if (
            "evaluate support answer quality" in system_prompt.lower()
            or "оцениваешь качество ответа поддержки" in system_prompt.lower()
        ):
            answer = _extract_block(user_prompt, "Answer draft:")
            length_score = min(1.0, max(0.2, len(answer) / 220.0))
            polite = 1.0 if any(k in answer.lower() for k in ["пожалуйста", "спасибо", "извин"]) else 0.7
            relevance = 0.9 if answer.strip() else 0.1
            return {
                "completeness": round(length_score, 2),
                "politeness": round(polite, 2),
                "relevance": round(relevance, 2),
                "notes": "Add one more actionable step if the answer is too short.",
            }

        return {}

    @overload
    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[StructuredOutputModel],
    ) -> StructuredOutputModel: ...

    @overload
    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: StructuredOutputSchema,
    ) -> BaseModel | dict[str, Any]:
        response = self.invoke_json(system_prompt, user_prompt)
        return _structured_response_to_schema(response, schema)

    def invoke_text(self, system_prompt: str, user_prompt: str) -> str:
        lower = user_prompt.lower()
        if "knowledge base context" in lower:
            kb_answer = _extract_first_kb_answer(user_prompt)
            if kb_answer:
                return f"Спасибо за вопрос. {kb_answer}"
            if "forgot password" in lower or "забыл пароль" in lower:
                return (
                    "Спасибо за обращение. Чтобы восстановить пароль, нажмите 'Забыли пароль' "
                    "на странице входа, укажите email и перейдите по ссылке из письма."
                )
            if "оплат" in lower or "payment" in lower:
                return (
                    "Спасибо за вопрос. Изменить способ оплаты можно в настройках биллинга: "
                    "добавьте новую карту и сделайте ее основной."
                )
            if "500" in lower or "ошибка" in lower:
                return (
                    "Спасибо, что сообщили. Попробуйте повторить действие после очистки кэша. "
                    "Если ошибка 500 сохраняется, проверьте страницу статуса и передайте request id в поддержку."
                )
            if "[]" in lower:
                return "Спасибо за вопрос. Сейчас мне не хватает данных в базе знаний для точного ответа."
            return (
                "Спасибо за обращение. Я подготовил ответ на основе базы знаний. "
                "Если нужно, уточните детали, и я помогу точнее."
            )
        if "rewrite the answer" in lower:
            current = _extract_block(user_prompt, "Current answer:")
            return (
                f"{current.strip()} Пожалуйста, если после этих шагов проблема останется, "
                "напишите номер тикета, и мы эскалируем обращение специалисту."
            )
        return "Спасибо за обращение. Я проверю информацию и помогу вам."


class GigaChatLLMClient(LLMClient):
    def __init__(self, settings: Settings):
        try:
            from langchain_gigachat import GigaChat  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("langchain-gigachat is not available") from exc

        if not settings.gigachat_credentials:
            raise RuntimeError("GIGACHAT_CREDENTIALS is required for gigachat provider")

        self.model = GigaChat(
            credentials=settings.gigachat_credentials,
            scope=settings.gigachat_scope,
            model=settings.gigachat_model,
            verify_ssl_certs=False,
            timeout=settings.timeout_seconds,
            top_p=0.9
        )

    def invoke_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self.model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        text = getattr(response, "content", "") if response else ""
        if isinstance(text, list):
            text = "".join(str(part) for part in text)
        return json.loads(_extract_json(text))

    @overload
    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[StructuredOutputModel],
    ) -> StructuredOutputModel: ...

    @overload
    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: StructuredOutputSchema,
    ) -> BaseModel | dict[str, Any]:
        structured_model = self.model.with_structured_output(schema)
        response = structured_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return _structured_response_to_schema(response, schema)

    def invoke_text(self, system_prompt: str, user_prompt: str) -> str:
        response = self.model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        content = getattr(response, "content", "") if response else ""
        if isinstance(content, list):
            return "".join(str(part) for part in content)
        return str(content)


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "gigachat":
        return GigaChatLLMClient(settings)
    if settings.llm_provider == "deepseek":
        raise NotImplementedError("DeepSeek provider is not implemented yet")
    return MockLLMClient()


def _structured_response_to_schema(
    response: Any,
    schema: StructuredOutputSchema,
) -> BaseModel | dict[str, Any]:
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        if isinstance(response, schema):
            return response
        if isinstance(response, BaseModel):
            return schema.model_validate(response.model_dump())
        return schema.model_validate(response)

    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        data = response.model_dump()
        if isinstance(data, dict):
            return data
    if hasattr(response, "dict"):
        data = response.dict()
        if isinstance(data, dict):
            return data
    raise TypeError(f"Structured output did not return a dict-like object: {type(response).__name__}")


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Model did not return valid JSON: {text}")
    return text[start : end + 1]


def _extract_block(text: str, marker: str) -> str:
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[idx + len(marker) :].strip()


def _extract_ticket_text(user_prompt: str) -> str:
    marker = "Ticket text:"
    start = user_prompt.find(marker)
    if start == -1:
        return user_prompt
    body = user_prompt[start + len(marker) :]
    stop_markers = [
        "\n\nAllowed category values:",
        "\n\nDynamic context:",
        "\n\nAnswer draft:",
    ]
    end_positions = [body.find(m) for m in stop_markers if body.find(m) != -1]
    if not end_positions:
        return body.strip()
    return body[: min(end_positions)].strip()


def _extract_first_kb_answer(user_prompt: str) -> str | None:
    marker = "Knowledge base context:"
    idx = user_prompt.find(marker)
    if idx == -1:
        return None
    context = user_prompt[idx + len(marker) :]
    match = re.search(r"Answer:\s*(.+?)(?:\n\[|$)", context, flags=re.DOTALL)
    if not match:
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        for line in lines:
            if line.startswith("[") and "]" in line:
                continue
            if line.lower().startswith("write the best possible reply"):
                break
            return line
        return None
    answer = match.group(1).strip()
    return answer if answer else None
