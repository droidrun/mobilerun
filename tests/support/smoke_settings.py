"""Smoke target identifiers. Importing these does not probe any service or device."""

import os

__all__ = ["SERIAL", "CLOUD_ID", "PROVIDER"]

SERIAL = os.getenv("MOBILERUN_SMOKE_SERIAL", "emulator-5554")
CLOUD_ID = os.getenv(
    "MOBILERUN_SMOKE_CLOUD_DEVICE_ID", "86f8d29f-9f59-4e42-ba02-1043c721774e"
)
PROVIDER = os.getenv("MOBILERUN_SMOKE_LLM_PROVIDER", "openai_oauth")
