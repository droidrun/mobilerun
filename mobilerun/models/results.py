"""Typed task results, independent of execution and transport."""

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ExecutionResult(BaseModel, Generic[T]):
    success: bool
    answer: str | None = None
    output: T | None = None
    steps: int | None = None
    task_id: str | None = None
    trajectory_path: str | None = None
    device: str


class ExpectationVerdict(BaseModel):
    passed: bool
    reason: str


class ExpectationResult(BaseModel):
    passed: bool
    condition: str
    reason: str
    task: ExecutionResult[ExpectationVerdict]
