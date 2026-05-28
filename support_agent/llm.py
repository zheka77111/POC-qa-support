from __future__ import annotations

import json
import re
from typing import Any, Mapping, TypeVar, overload

from langchain.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from support_agent.config import Settings
from support_agent.state import Category

def build_chat_model(settings: Settings) -> BaseChatModel:
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