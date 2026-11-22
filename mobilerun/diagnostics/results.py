from dataclasses import dataclass
from enum import Enum


class Status(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    detail: str = ""
