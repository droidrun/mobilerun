"""Isolated fixtures for explicitly requested real-service checks."""

import asyncio
import importlib.util
import json
import os
import shutil
import subprocess

import pytest

SERIAL = os.getenv("MOBILERUN_SMOKE_SERIAL", "emulator-5554")
CLOUD_ID = os.getenv(
    "MOBILERUN_SMOKE_CLOUD_DEVICE_ID", "86f8d29f-9f59-4e42-ba02-1043c721774e"
)
PROVIDER = os.getenv("MOBILERUN_SMOKE_LLM_PROVIDER", "openai_oauth")


@pytest.fixture(autouse=True)
def android_sdk_for_smoke(monkeypatch, real_android_sdk, isolate_user_configuration):
    monkeypatch.setenv("ANDROID_HOME", real_android_sdk)


def pytest_collection_modifyitems(items):
    for item in items:
        if "/smoke/" in str(item.path):
            item.add_marker(pytest.mark.smoke)


def check_requirements(auth_path):
    checks = {}
    key = os.getenv("MOBILERUN_API_KEY")
    checks["Cloud API key present"] = bool(key)
    checks["mobilerun_sdk importable"] = (
        importlib.util.find_spec("mobilerun_sdk") is not None
    )
    checks["Cloud API key accepted"] = False
    checks["Cloud device exists"] = False
    checks["Cloud device ready"] = False
    if key and checks["mobilerun_sdk importable"]:

        async def lookup():
            from mobilerun_sdk import AsyncMobilerun

            from mobilerun.config.settings import get_settings

            async with AsyncMobilerun(
                api_key=key,
                base_url=get_settings().MOBILERUN_CLOUD_BASE_URL,
                timeout=15.0,
            ) as client:
                return await client.devices.list()

        try:
            response = asyncio.run(lookup())
            checks["Cloud API key accepted"] = True
            item = next(
                (item for item in response.items if str(item.id) == CLOUD_ID), None
            )
            checks["Cloud device exists"] = item is not None
            checks["Cloud device ready"] = item is not None and item.state == "ready"
        except Exception:
            pass
    adb = shutil.which("adb")
    checks["adb on PATH"] = bool(adb)
    checks["Local serial online"] = False
    if adb:
        try:
            result = subprocess.run(
                [adb, "-s", SERIAL, "get-state"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            checks["Local serial online"] = (
                result.returncode == 0 and result.stdout.strip() == "device"
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    if PROVIDER.endswith("_oauth"):
        try:
            profile = json.loads(auth_path.read_text())
            slot = {
                "openai_oauth": "openaiOauth",
                "anthropic_oauth": "anthropicOauth",
                "gemini_oauth": "geminiOauth",
                "grok_oauth": "grokOauth",
            }.get(PROVIDER)
            checks["LLM credentials"] = bool(slot and profile.get(slot))
        except (OSError, ValueError):
            checks["LLM credentials"] = False
    else:
        checks["LLM credentials"] = bool(os.getenv(PROVIDER.upper() + "_API_KEY"))
    return checks


@pytest.fixture
def requirements(real_env, copied_auth_profiles):
    return check_requirements(copied_auth_profiles)


@pytest.fixture
def cloud_ready(requirements):
    if not all(
        requirements[name]
        for name in (
            "Cloud API key present",
            "mobilerun_sdk importable",
            "Cloud API key accepted",
            "Cloud device exists",
            "Cloud device ready",
        )
    ):
        pytest.skip("Cloud requirement missing; run test_00_requirements.py")


@pytest.fixture
def local_ready(requirements):
    if not requirements["Local serial online"]:
        pytest.skip("Local requirement missing; run test_00_requirements.py")
