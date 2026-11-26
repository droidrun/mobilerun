"""Minimal cloud function that sends a static result to a webhook.

Deploy (paid, device-free endpoint):
    .venv/bin/python tests/demos/webhook_static_demo.py
Fire it:
    mobilerun sandbox fire webhook-static-demo
The result arrives at the webhook as `workflow.step.emit` with the return value in `data.result`.
"""

from pydantic import BaseModel

from mobilerun.sandbox import cloud, deploy


class Input(BaseModel):
    pass


class Output(BaseModel):
    message: str
    source: str
    ok: bool


@cloud("webhook-static-demo", webhook="0ab5e848-f7aa-4119-80a0-57ffcd0c2dea")
def main(input: Input) -> Output:
    return Output(message="hello from mobilerun", source="webhook-static-demo", ok=True)


if __name__ == "__main__":
    deploy(__file__, device_free=True, yes=True)
