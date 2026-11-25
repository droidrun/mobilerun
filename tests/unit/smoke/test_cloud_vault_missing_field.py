"""Updating a missing credential returns safe support diagnostics; creates nothing."""

import asyncio
import traceback
from uuid import uuid4

import pytest


@pytest.mark.smoke_cloud
def test_missing_vault_field_preserves_http_status(request):
    request.getfixturevalue("real_env")
    from mobilerun.core.credentials import CloudCredentials
    from mobilerun.errors import CloudResourceNotFound

    async def run():
        vault = CloudCredentials(billing="no_commerce")
        value = uuid4().hex
        try:
            with pytest.raises(CloudResourceNotFound) as caught:
                await vault.update(
                    "ai.mobilerun.smoke.test", uuid4().hex, "password", value
                )
            assert caught.value.status == 404 and caught.value.value_safe
            assert value not in "".join(traceback.format_exception(caught.value))
        finally:
            await vault.close()

    asyncio.run(run())
