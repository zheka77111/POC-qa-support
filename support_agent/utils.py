import json
from typing import Callable
from rich.table import Table
from rich.console import Console
from langchain.agents.middleware import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    wrap_model_call,
)
import pandas as pd
from langchain.messages import AIMessage
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

console = Console()
def show_prompt(prompt_text: str, title: str = "Prompt", border_style: str = "white"):
    """Display a prompt with rich formatting and XML tag highlighting.

    Args:
        prompt_text: The prompt string to display
        title: Title for the panel (default: "Prompt")
        border_style: Border color style (default: "blue")
    """
    # Create a formatted display of the prompt
    formatted_text = Text(prompt_text)
    formatted_text.highlight_regex(r"<[^>]+>", style="bold blue")  # Highlight XML tags
    formatted_text.highlight_regex(
        r"##[^#\n]+", style="bold magenta"
    )  # Highlight headers
    formatted_text.highlight_regex(
        r"###[^#\n]+", style="bold cyan"
    )  # Highlight sub-headers

    # Display in a panel for better presentation
    console.print(
        Panel(
            formatted_text,
            title=f"[bold green]{title}[/bold green]",
            border_style=border_style,
            padding=(1, 2),
        )
    )


    """Utility functions for displaying messages and prompts in Jupyter notebooks."""



console = Console()


def format_message_content(message):
    """Convert message content to displayable string."""
    parts = []
    tool_calls_processed = False

    # Handle main content
    if isinstance(message.content, str):
        parts.append(message.content)
    elif isinstance(message.content, list):
        # Handle complex content like tool calls (Anthropic format)
        for item in message.content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "tool_use":
                    parts.append(f"\n🔧 Tool Call: {item.get('name', 'N/A')}")
                    parts.append(
                        f"   Args: {json.dumps(item.get('input', {}), indent=2, ensure_ascii=False)}"
                    )
                    parts.append(f"   ID: {item.get('id', 'N/A')}")
                    tool_calls_processed = True
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
    else:
        parts.append(str(message.content))

    # Handle tool calls attached to the message (OpenAI format) - only if not already processed
    if (
        not tool_calls_processed
        and hasattr(message, "tool_calls")
        and message.tool_calls
    ):
        for tool_call in message.tool_calls:
            parts.append(f"\n🔧 Tool Call: {tool_call['name']}")
            parts.append(f"   Args: {json.dumps(tool_call['args'], indent=2, ensure_ascii=False)}")
            parts.append(f"   ID: {tool_call['id']}")

    return "\n".join(parts)


def format_messages(messages):
    """Format and display messages and node events with Rich formatting."""

    def _iter_events(obj):
        if obj is None:
            return
        if isinstance(obj, dict):
            events = obj.get("events")
            if isinstance(events, list):
                for event in events:
                    if isinstance(event, dict) and event.get("node") and event.get("event"):
                        yield event
            for value in obj.values():
                yield from _iter_events(value)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                yield from _iter_events(item)

    def _iter_message_like(obj):
        if obj is None:
            return
        if hasattr(obj, "content"):
            yield obj
            return
        if isinstance(obj, dict):
            if isinstance(obj.get("messages"), list):
                for item in obj["messages"]:
                    yield from _iter_message_like(item)
            else:
                for value in obj.values():
                    yield from _iter_message_like(value)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                yield from _iter_message_like(item)
            return
        yield obj

    for event in _iter_events(messages):
        node = event.get("node", "unknown")
        event_name = event.get("event", "unknown")
        payload = event.get("payload", {})
        timestamp = event.get("timestamp")

        lines = [f"event: {event_name}"]
        if timestamp:
            lines.append(f"timestamp: {timestamp}")
        if payload:
            lines.append("payload:")
            lines.append(json.dumps(payload, ensure_ascii=False, indent=2))

        console.print(
            Panel(
                "\n".join(lines),
                title=f"Node: {node}",
                border_style="magenta",
            )
        )

    for m in _iter_message_like(messages):
        if hasattr(m, "content"):
            msg_type = m.__class__.__name__.replace("Message", "")
            content = format_message_content(m)

            if msg_type == "Human":
                console.print(Panel(Markdown(content), title="🧑 Human", border_style="blue"))
            elif msg_type == "AI" or msg_type == "model":
                console.print(Panel(Markdown(content), title="🤖 Assistant", border_style="green"))
            elif msg_type == "Tool" or msg_type == "tools":
                console.print(Panel(Markdown(content), title="🔧 Tool Output", border_style="yellow"))
            else:
                console.print(Panel(Markdown(content), title=f"📝 {msg_type}", border_style="white"))
        else:
            continue


def format_message(messages):
    """Alias for format_messages for backward compatibility."""
    return format_messages(messages)


def run_dialog(agent: CompiledStateGraph, mas_array: list[dict[str, list[dict[str, str]]]], config: dict) -> None:
    """
    Run a dialog with the agent using a series of messages.

    Args:
        agent: The compiled state graph agent
        mas_array: List of message dictionaries
        thread_id: Thread identifier for the conversation
    
    """
    for mes in mas_array:
        # config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 350 }
        console.print(Panel(mes['messages'][0]['content'], title="🧑 Human", border_style="white"))
        for chunk in agent.stream(mes, config):
            if 'model' in chunk:
                format_messages(chunk['model']['messages'])
            if 'tools' in chunk:
                format_messages(chunk['tools']['messages'])


def print_eval(df:pd.DataFrame) -> None:
    """
    Печатает DataFrame в виде таблицы с помощью rich
    """
    c = Console()
    t = Table(*df.columns.to_list(), show_lines=True, style = "dim")
    for _, row in df.iterrows():
        t.add_row(*[str(val) for val in row.tolist()])
    c.print(t)





def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None

def _latest_user_query(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_text(message)
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
