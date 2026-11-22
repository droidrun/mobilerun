"""Free real-service checks for cloud API key validation."""

import asyncio
import os

import pytest

from mobilerun.clients.cloud.api_keys import validate_api_key
from mobilerun.errors.cloud import CloudAuthenticationError


def test_real_api_key_is_valid(real_env):
    if not os.environ.get("MOBILERUN_API_KEY"):
        pytest.skip("MOBILERUN_API_KEY is not set")
    asyncio.run(validate_api_key(os.environ["MOBILERUN_API_KEY"], billing="included"))


def test_bad_api_key_is_invalid(real_env):
    with pytest.raises(CloudAuthenticationError):
        asyncio.run(
            validate_api_key("obviously-invalid-mobilerun-api-key", billing="included")
        )
