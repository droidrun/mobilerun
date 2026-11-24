from datetime import datetime

from mobilerun_functions import Context
from pydantic import BaseModel

from mobilerun.sandbox import cloud


class Input(BaseModel):
    app_package: str | None = None


class Output(BaseModel):
    device_id: str
    captured_at: datetime
    screenshot_path: str
    ui_node_count: int
    top_texts: list[str]


@cloud("playground-screenshot-ui-test")
def main(input: Input, ctx: Context) -> Output:
    import os
    import time
    from datetime import datetime, timezone

    started = time.monotonic()
    ctx.log.info(
        f"Run6 started: run_id={ctx.run.run_id} attempt={ctx.run.attempt} "
        f"mode={ctx.run.mode} target={ctx.run.device_target} app_package={input.app_package}"
    )

    t = time.monotonic()
    ctx.log.info(f"Connecting to device {ctx.run.device_id}")
    device = ctx.device()
    ctx.log.info(f"Connected to {device.device_id} in {time.monotonic() - t:.2f}s")

    if input.app_package:
        t = time.monotonic()
        ctx.log.info(f"Opening app {input.app_package}")
        device.start_app(input.app_package)
        ctx.log.info(f"App opened in {time.monotonic() - t:.2f}s")
    else:
        ctx.log.info("No app_package given, reading current screen")

    t = time.monotonic()
    ctx.log.info("Taking screenshot")
    screenshot_path = device.screenshot()
    captured_at = datetime.now(timezone.utc)
    size_kb = os.path.getsize(screenshot_path) / 1024
    ctx.log.info(
        f"Screenshot saved: {screenshot_path} ({size_kb:.0f} KB) in {time.monotonic() - t:.2f}s"
    )

    t = time.monotonic()
    ctx.log.info("Reading UI tree")
    nodes = device.ui().nodes
    texts = [node["text"] for node in nodes if node.get("text")]
    ctx.log.info(
        f"UI tree read in {time.monotonic() - t:.2f}s: {len(nodes)} nodes, {len(texts)} with text"
    )
    if not texts:
        ctx.log.warning("No visible text found on screen")

    ctx.log.info(f"Run finished in {time.monotonic() - started:.2f}s")
    return Output(
        device_id=device.device_id,
        captured_at=captured_at,
        screenshot_path=screenshot_path,
        ui_node_count=len(nodes),
        top_texts=texts[:10],
    )
