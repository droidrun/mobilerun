import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
import pytest
from mobilerun_sdk import AsyncMobilerun

from mobilerun.clients.cloud import devices, proxies, secrets
from mobilerun.errors import CloudValidationError
from mobilerun.models.billing import Billing


@pytest.mark.parametrize(
    "operation", ["device", "proxy", "save", "create_field", "update"]
)
def test_sdk_request_options_never_log_secrets(monkeypatch, caplog, operation):
    secret = "sdk-sentinel-password-987654"
    caplog.set_level(logging.DEBUG, logger="mobilerun_sdk._base_client")
    sdk = AsyncMobilerun(
        api_key="test",
        base_url="https://example.test",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(400, json={"message": "rejected"})
            )
        ),
    )

    @asynccontextmanager
    async def client(*args, **kwargs):
        yield sdk

    monkeypatch.setattr(proxies, "cloud_client", client)

    async def run():
        try:
            with pytest.raises(CloudValidationError):
                if operation == "device":
                    await devices.create(
                        client=sdk,
                        billing=Billing.PAID,
                        proxy={
                            "socks5": {
                                "host": "example.test",
                                "port": 1080,
                                "password": secret,
                            }
                        },
                    )
                elif operation == "proxy":
                    await proxies.connect_device(
                        "device",
                        {
                            "socks5": {
                                "host": "example.test",
                                "port": 1080,
                                "password": secret,
                            }
                        },
                        billing=Billing.NO_COMMERCE,
                    )
                elif operation == "save":
                    await secrets.save(
                        sdk,
                        "pkg",
                        "name",
                        {"password": secret},
                        billing=Billing.NO_COMMERCE,
                    )
                elif operation == "create_field":
                    await secrets.create_field(
                        sdk,
                        "pkg",
                        "name",
                        "password",
                        secret,
                        billing=Billing.NO_COMMERCE,
                    )
                else:
                    await secrets.update(
                        sdk,
                        "pkg",
                        "name",
                        "password",
                        secret,
                        billing=Billing.NO_COMMERCE,
                    )
        finally:
            await sdk.close()

    asyncio.run(run())
    assert any("Request options" in r.getMessage() for r in caplog.records)
    assert all(secret not in repr(r.__dict__) for r in caplog.records)
