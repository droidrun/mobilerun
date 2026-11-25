"""Staging-aware variants of the shared sandbox fakes."""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from mobilerun.sandbox import clients
from mobilerun.sandbox import deployment as deploy
from tests.support.sandbox_fakes import InitialSandboxCloud as BaseInitialCloud
from tests.support.sandbox_fixtures import changed as base_changed
from tests.support.sandbox_fixtures import env as base_env

__all__ = [
    "base_changed",
    "base_env",
    "changed",
    "env",
    "initial_cloud",
    "InitialSandboxCloud",
]


class InitialSandboxCloud(BaseInitialCloud):
    async def prepare_action_payloads(self, document, *, billing):
        return await clients.prepare_action_payloads(
            document, billing=billing, client=self.sdk
        )

    async def create_document(self, document, key, *, billing, action_payloads):
        self.document = deepcopy(document)
        return await clients.create_document(
            document,
            key,
            billing=billing,
            client=self.sdk,
            action_payloads=action_payloads,
        )


@pytest.fixture
def env(base_env):
    base_env.fake.prepare_action_payloads = AsyncMock(side_effect=_action_payloads)
    return base_env


def _action_payloads(document, **kwargs):
    return [
        dict(
            catalog_entry_id=(
                "webhooks-catalog" if step["uses"] == "webhooks.send" else "catalog"
            ),
            name=step["name"],
            params=step["with"],
        )
        for step in document["steps"]
    ]


@pytest.fixture
def changed(base_changed):
    base_changed.cloud.prepare_action_payloads = AsyncMock(side_effect=_action_payloads)
    return base_changed


@pytest.fixture
def initial_cloud(env, monkeypatch):
    env.cloud = InitialSandboxCloud(
        deploy.build_document(env.artifact, "f", 1, device=None)
    )
    monkeypatch.setattr(deploy, "clients", env.cloud)
    return env
