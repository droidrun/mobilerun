"""Verify a real test POST, cleaning up the new endpoint and receiver token."""

import json
import time
from contextlib import ExitStack
from uuid import UUID

import httpx
import pytest
from click.testing import CliRunner

from mobilerun.cli.app import cli
from mobilerun.utils.redaction import secret_scope


@pytest.mark.smoke_cloud
def test_webhook_cli_lifecycle(real_env):
    runner = CliRunner()
    with (
        httpx.Client(base_url="https://webhook.site", timeout=15) as receiver,
        ExitStack() as cleanup,
    ):

        def request(method, path, **kwargs):
            try:
                response = receiver.request(method, path, **kwargs)
            except httpx.HTTPError:
                pytest.fail("Webhook.site request did not complete", pytrace=False)
            if not response.is_success:
                pytest.fail(
                    f"Webhook.site {method} failed: HTTP {response.status_code}",
                    pytrace=False,
                )
            return response

        token = str(
            UUID(
                request(
                    "POST", "/token", json={"expiry": 300, "request_limit": 5}
                ).json()["uuid"]
            )
        )
        cleanup.enter_context(secret_scope(token))

        def delete_token():
            request("DELETE", f"/token/{token}")
            print("receiver deleted")

        cleanup.callback(delete_token)
        url = f"https://webhook.site/{token}"
        result = runner.invoke(cli, ["webhook", "register", url, "--json"])
        if result.exit_code:
            pytest.fail("Webhook registration failed", pytrace=False)
        registered = json.loads(result.stdout)
        if registered["created"] is not True:
            pytest.fail("Endpoint not newly created; refusing deletion", pytrace=False)
        endpoint_id = registered["id"]

        def delete_endpoint():
            if runner.invoke(
                cli, ["webhook", "delete", endpoint_id, "--yes"]
            ).exit_code:
                pytest.fail(f"Cleanup failed for endpoint {endpoint_id}", pytrace=False)
            print(f"Deleted endpoint: {endpoint_id}")

        cleanup.callback(delete_endpoint)
        if not registered.get("secret"):
            pytest.fail(
                "Created endpoint did not return a signing secret", pytrace=False
            )
        listed = runner.invoke(cli, ["webhook", "list", "--json"])
        assert listed.exit_code == 0
        assert any(row["id"] == endpoint_id for row in json.loads(listed.stdout))
        ping = runner.invoke(cli, ["webhook", "ping", endpoint_id])
        assert ping.exit_code == 0 and "Delivered: HTTP 200" in ping.output
        # The test API returns after delivery; webhook.site indexes captures asynchronously.
        for attempt in range(10):
            requests = request(
                "GET",
                f"/token/{token}/requests",
                params={"page": 1, "per_page": 5, "_": attempt},
            ).json()["data"]
            if any(r["method"] == "POST" for r in requests):
                return
            time.sleep(1)
        pytest.fail(
            f"No POST indexed for the fresh receiver ({len(requests)} captures)",
            pytrace=False,
        )
