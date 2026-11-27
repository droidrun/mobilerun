"""Planned Mobilerun sandbox App example (commands await sandbox-api release)."""

from pydantic import BaseModel

import mobilerun

app = mobilerun.sandbox.App("example-counter-tests")


class Input(BaseModel):
    n: int = 3


class Output(BaseModel):
    count: int


@app.task(device="86f8d29f-0000-0000-0000-000000000000", timeout=600)
def count_to(input: Input, ctx: mobilerun.sandbox.Context) -> Output:
    phone = ctx.device()
    phone.apps.start("ai.droidrun.counter")
    return Output(count=input.n)


@app.local_entrypoint()
def main(n: int = 3) -> None:
    print(count_to.remote(Input(n=n)))
