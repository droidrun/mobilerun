from __future__ import annotations

import base64
from typing import Sequence
from urllib.parse import urlparse

import httpx
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, ImageBlock, TextBlock


def _extract_last_image(messages: Sequence[ChatMessage]) -> ImageBlock | None:
    for message in reversed(messages):
        for block in reversed(message.blocks):
            if isinstance(block, ImageBlock):
                return block
    return None


def _build_prompt(messages: Sequence[ChatMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        text_parts = [
            block.text.strip()
            for block in message.blocks
            if isinstance(block, TextBlock) and block.text.strip()
        ]
        if text_parts:
            parts.append(f"{message.role.value.upper()}:\n" + "\n".join(text_parts))
    return "\n\n".join(parts).strip()


def _resolve_vlm_url(api_base: str | None) -> str:
    raw = (api_base or "https://api.minimax.io/v1").strip()
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else "https://api.minimax.io"
    return f"{origin}/v1/coding_plan/vlm"


def _image_block_to_data_url(block: ImageBlock) -> str:
    if block.url:
        return str(block.url)
    image_bytes = block.resolve_image().read()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{block.image_mimetype};base64,{encoded}"


async def aminimax_vision_chat(
    llm,
    messages: Sequence[ChatMessage],
    timeout: float = 90.0,
) -> ChatResponse:
    image_block = _extract_last_image(messages)
    if image_block is None:
        raise ValueError("MiniMax vision call requires an image block.")

    prompt = _build_prompt(messages)
    if not prompt:
        raise ValueError("MiniMax vision call requires text context.")

    api_key = getattr(llm, "api_key", None)
    if not api_key:
        raise ValueError("MiniMax vision call requires an API key.")

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            _resolve_vlm_url(getattr(llm, "api_base", None)),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "MM-API-Source": "Droidrun",
            },
            json={
                "prompt": prompt,
                "image_url": _image_block_to_data_url(image_block),
            },
        )

    response.raise_for_status()
    payload = response.json()
    base_resp = payload.get("base_resp") if isinstance(payload, dict) else None
    if isinstance(base_resp, dict) and base_resp.get("status_code") not in (None, 0):
        status_code = base_resp.get("status_code")
        status_msg = (base_resp.get("status_msg") or "").strip()
        raise ValueError(
            f"MiniMax vision API error ({status_code})"
            + (f": {status_msg}" if status_msg else "")
        )

    content = payload.get("content") if isinstance(payload, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("MiniMax vision API returned no content.")

    return ChatResponse(
        message=ChatMessage(role="assistant", blocks=[TextBlock(text=content.strip())]),
        raw=payload,
        additional_kwargs={},
    )
