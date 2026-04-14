import logging
from io import BytesIO
from pathlib import Path
from typing import Union

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock
from PIL import Image

logger = logging.getLogger("droidrun")


# ============================================================================
# CONVERSION TO CHATMESSAGE (call right before LLM)
# ============================================================================


def _ensure_image_bytes(image_source: Union[str, Path, Image.Image, bytes]) -> bytes:
    """Convert image to bytes."""
    if isinstance(image_source, bytes):
        return image_source
    if isinstance(image_source, (str, Path)):
        image = Image.open(image_source)
    elif isinstance(image_source, Image.Image):
        image = image_source
    else:
        raise ValueError(f"Unsupported image type: {type(image_source)}")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def to_chat_messages(messages: list[dict]) -> list[ChatMessage]:
    """
    Convert dict messages to ChatMessage list.

    Args:
        messages: List of message dicts

    Returns:
        List of ChatMessage objects
    """
    chat_messages = []

    for msg in messages:
        blocks = []
        for item in msg.get("content", []):
            if "text" in item:
                blocks.append(TextBlock(text=item["text"]))
            elif "image" in item:
                image_bytes = _ensure_image_bytes(item["image"])
                blocks.append(ImageBlock(image=image_bytes))

        chat_messages.append(ChatMessage(role=msg["role"], blocks=blocks))

    return chat_messages


# ============================================================================
# MESSAGE UTILITIES
# ============================================================================


def has_content(message: ChatMessage) -> bool:
    for block in message.blocks:
        if isinstance(block, TextBlock) and block.text and block.text.strip():
            return True
        if isinstance(block, ImageBlock) and block.image:
            return True
    return False


def filter_empty_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    return [msg for msg in messages if has_content(msg)]


def limit_history(
    messages: list[ChatMessage], max_messages: int, preserve_first: bool = True
) -> list[ChatMessage]:
    if len(messages) <= max_messages:
        return messages

    if preserve_first and messages:
        first = messages[0]
        tail = messages[-max_messages + 1 :]
        if first not in tail:
            return [first] + tail
        return tail

    return messages[-max_messages:]
