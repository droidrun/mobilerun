import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from click.testing import CliRunner

from mobilerun.cli.phone_number import group
from mobilerun.clients.cloud import numbers
from mobilerun.models.billing import Billing


def test_number_create_without_country_reaches_api():
    endpoint = SimpleNamespace(
        create=AsyncMock(
            return_value=SimpleNamespace(
                data=SimpleNamespace(number_id="n1", state="active", checkout_url=None)
            )
        )
    )
    asyncio.run(
        numbers.create(billing=Billing.PAID, client=SimpleNamespace(numbers=endpoint))
    )
    endpoint.create.assert_awaited_once_with(billing_preference="included")


def test_known_validation_code_is_safe_and_actionable():
    from mobilerun.clients.cloud.errors import cloud_error

    exc = SimpleNamespace(
        status_code=422,
        response=None,
        request_id=None,
        body={"code": "unsupported_country", "message": "SECRET"},
    )
    error = cloud_error(exc, "Create number")
    assert "mobilerun number countries" in str(error)
    assert "SECRET" not in str(error)


def test_unknown_validation_code_does_not_leak():
    from mobilerun.clients.cloud.errors import cloud_error

    exc = SimpleNamespace(
        status_code=422,
        response=None,
        request_id=None,
        body={"code": "SECRET", "message": "SECRET"},
    )
    assert "SECRET" not in str(cloud_error(exc, "Create number"))


def test_known_device_validation_code_is_safe():
    from mobilerun.clients.cloud.errors import cloud_error

    exc = SimpleNamespace(
        status_code=422,
        response=None,
        request_id=None,
        body={"code": "DEVICE_TYPE_UNAVAILABLE", "message": "SECRET"},
    )
    result = str(cloud_error(exc, "Create cloud phone"))
    assert "cloud phone type is unavailable" in result
    assert "SECRET" not in result


def test_noninteractive_create_forwards_missing_country(monkeypatch):
    monkeypatch.setattr(group, "is_interactive", lambda: False)
    monkeypatch.setattr(group, "confirm_action_async", AsyncMock())
    created = []

    async def phone(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(id="n1")

    monkeypatch.setattr(group, "PhoneNumber", phone)
    monkeypatch.setattr(
        group.cloud_numbers,
        "retrieve",
        AsyncMock(
            return_value=SimpleNamespace(
                data=SimpleNamespace(state="active", phone_number="+1")
            )
        ),
    )
    result = CliRunner().invoke(group.number, ["create", "--billing", "paid", "--yes"])
    assert result.exit_code == 0, result.output
    assert created[0]["country"] is None


def test_number_create_prompts_for_supported_country(monkeypatch):
    monkeypatch.setattr(group, "is_interactive", lambda: True)
    monkeypatch.setattr(
        group.cloud_numbers,
        "countries",
        AsyncMock(
            return_value=SimpleNamespace(
                data=SimpleNamespace(
                    items=[
                        SimpleNamespace(
                            country="US", name="United States", in_stock=True
                        )
                    ]
                )
            )
        ),
    )
    monkeypatch.setattr(
        group, "select_prompt", lambda message, choices: choices[0].value
    )
    monkeypatch.setattr(group, "confirm_action_async", AsyncMock())
    created = []

    async def phone(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(id="n1")

    monkeypatch.setattr(group, "PhoneNumber", phone)
    monkeypatch.setattr(
        group.cloud_numbers,
        "retrieve",
        AsyncMock(
            return_value=SimpleNamespace(
                data=SimpleNamespace(state="active", phone_number="+1")
            )
        ),
    )
    result = CliRunner().invoke(group.number, ["create", "--billing", "paid", "--yes"])
    assert result.exit_code == 0, result.output
    assert created[0]["country"] == "US"
    assert created[0]["checkout"] == "browser"
