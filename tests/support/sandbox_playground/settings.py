from datetime import datetime

from mobilerun_functions import Context
from pydantic import BaseModel

from mobilerun.sandbox import cloud


class Input(BaseModel):
    wait_s: float = 2.0


class Output(BaseModel):
    app_id: str
    settings_open: bool
    ui_node_count: int
    texts: list[str]
    screenshot: str | None
    captured_at: datetime


@cloud("playground-open-settings")
def main(input: Input, ctx: Context) -> Output:
    import base64
    import io
    import time
    from datetime import datetime, timezone

    from PIL import Image

    started = time.monotonic()
    device = ctx.device()

    def capture():
        t = time.monotonic()
        path = device.screenshot()
        captured = datetime.now(timezone.utc)
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
            for quality in (70, 60, 50, 40, 30):
                data = io.BytesIO()
                image.save(data, format="JPEG", quality=quality, optimize=True)
                encoded = base64.b64encode(data.getvalue()).decode("ascii")
                if len(encoded) <= 240000:
                    ctx.log.info(
                        f"Screenshot captured and compressed in {time.monotonic() - t:.2f}s"
                    )
                    return encoded, captured
        ctx.log.warning("Screenshot exceeds JPEG budget; omitted")
        ctx.partial("screenshot exceeds JPEG budget")
        return None, captured

    try:
        t = time.monotonic()
        device.key("home")
        ctx.log.info(f"Home in {time.monotonic() - t:.2f}s")
        t = time.monotonic()
        device.start_app("com.android.settings")
        ctx.log.info(f"Settings opened in {time.monotonic() - t:.2f}s")
        t = time.monotonic()
        time.sleep(max(0, input.wait_s))
        ctx.log.info(f"Settled in {time.monotonic() - t:.2f}s")
        screenshot, captured = capture()
        t = time.monotonic()
        nodes = device.ui().nodes
        texts = [node["text"] for node in nodes if node.get("text")][:20]
        ctx.log.info(f"UI read: {len(nodes)} nodes in {time.monotonic() - t:.2f}s")
        t = time.monotonic()
        app_id = device.current_app_id() or ""
        ctx.log.info(f"Current app: {app_id} in {time.monotonic() - t:.2f}s")
        return Output(
            app_id=app_id,
            settings_open=app_id == "com.android.settings",
            ui_node_count=len(nodes),
            texts=texts,
            screenshot=screenshot,
            captured_at=captured,
        )
    finally:
        t = time.monotonic()
        device.key("home")
        ctx.log.info(
            f"Returned home in {time.monotonic() - t:.2f}s; total {time.monotonic() - started:.2f}s"
        )
