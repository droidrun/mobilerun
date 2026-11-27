from mobilerun_functions import Context
from pydantic import BaseModel

from mobilerun.sandbox import cloud


class Input(BaseModel):
    name: str | None = None


class Output(BaseModel):
    message: str
    version: int


@cloud("playground-ping")
def main(input: Input, ctx: Context) -> Output:
    return Output(message="pong", version=1)
