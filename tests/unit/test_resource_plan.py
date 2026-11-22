import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.clients.cloud import mailboxes, numbers
from mobilerun.core.mailbox.cloud import CloudMailbox
from mobilerun.core.phone_number.cloud import CloudPhoneNumber
from mobilerun.errors.cloud import BillingError


@pytest.mark.parametrize("resource", [numbers, mailboxes])
@pytest.mark.parametrize(
    "billing,plan,expected",
    [
        ("included", None, "included_only"),
        ("paid", None, "included"),
        ("included", "auto", "included_only"),
        ("paid", "auto", "included"),
        ("included", "included", "included_only"),
        ("paid", "included", "included_only"),
        ("paid", "rent", "rent"),
    ],
)
def test_create_preference(resource, billing, plan, expected):
    data = SimpleNamespace(
        number_id="num",
        state="active",
        checkout_url=None,
        id="box",
        status="active",
        address="x@example.com",
    )
    create = AsyncMock(return_value=SimpleNamespace(data=data))
    client = SimpleNamespace(
        numbers=SimpleNamespace(create=create), mailboxes=SimpleNamespace(create=create)
    )
    kwargs = {} if plan is None else {"plan": plan}
    asyncio.run(resource.create(billing=billing, client=client, **kwargs))
    assert create.await_args.kwargs["billing_preference"] == expected


@pytest.mark.parametrize("resource", [numbers, mailboxes])
@pytest.mark.parametrize("billing", ["included", "no_commerce"])
def test_rent_requires_paid(resource, billing):
    create = AsyncMock()
    client = SimpleNamespace(
        numbers=SimpleNamespace(create=create), mailboxes=SimpleNamespace(create=create)
    )
    with pytest.raises(BillingError) as exc:
        asyncio.run(resource.create(billing=billing, client=client, plan="rent"))
    assert exc.value.code == "invalid_billing"
    assert exc.value.required_billing == "paid"
    create.assert_not_awaited()


@pytest.mark.parametrize("resource", [numbers, mailboxes])
def test_invalid_plan_rejected_before_sdk(resource):
    create = AsyncMock()
    client = SimpleNamespace(
        numbers=SimpleNamespace(create=create), mailboxes=SimpleNamespace(create=create)
    )
    with pytest.raises(ValueError):
        asyncio.run(resource.create(billing="paid", client=client, plan="invalid"))
    create.assert_not_awaited()


@pytest.mark.parametrize("resource", [numbers, mailboxes])
@pytest.mark.parametrize("field", ["extra_body", "extra_query"])
def test_billing_preference_cannot_be_smuggled(resource, field):
    create = AsyncMock()
    client = SimpleNamespace(
        numbers=SimpleNamespace(create=create), mailboxes=SimpleNamespace(create=create)
    )
    with pytest.raises(BillingError) as exc:
        asyncio.run(
            resource.create(
                billing="paid",
                client=client,
                plan="rent",
                **{field: {"billing_preference": "included"}},
            )
        )
    assert exc.value.code == "billing_override"
    create.assert_not_awaited()


@pytest.mark.parametrize(
    "cls,resource", [(CloudPhoneNumber, numbers), (CloudMailbox, mailboxes)]
)
@pytest.mark.parametrize("plan", ["auto", "included", "rent"])
def test_core_forwards_plan(monkeypatch, cls, resource, plan):
    data = SimpleNamespace(
        number_id="num",
        state="active",
        checkout_url=None,
        id="box",
        status="active",
        address="x@example.com",
    )
    create = AsyncMock(return_value=SimpleNamespace(data=data))
    monkeypatch.setattr(resource, "create", create)

    async def run():
        return await cls(billing="paid", plan=plan)

    asyncio.run(run())
    assert create.await_args.kwargs["plan"] == plan


@pytest.mark.parametrize(
    "cls,resource", [(CloudPhoneNumber, numbers), (CloudMailbox, mailboxes)]
)
@pytest.mark.parametrize(
    "status,expected", [("available", 3), ("exhausted", 0), ("not_included", 0)]
)
def test_included_available(monkeypatch, cls, resource, status, expected):
    remaining = 3 if status != "exhausted" else 0
    capacity = AsyncMock(
        return_value=SimpleNamespace(
            data=SimpleNamespace(
                status=status, remaining=remaining, included_remaining=99
            )
        )
    )
    monkeypatch.setattr(resource, "capacity", capacity)
    args = ("US",) if cls is CloudPhoneNumber else ()
    client = object()
    assert (
        asyncio.run(cls.included_available(*args, billing="no_commerce", client=client))
        == expected
    )
    capacity.assert_awaited_once_with(*args, billing="no_commerce", client=client)


def test_resource_plan_public_export():
    from mobilerun.models import ResourcePlan

    assert [p.value for p in ResourcePlan] == ["auto", "included", "rent"]


@pytest.mark.parametrize("cls", [CloudPhoneNumber, CloudMailbox])
def test_constructor_rejects_invalid_plan_eagerly(cls):
    with pytest.raises(ValueError):
        cls(billing="paid", plan="invalid")


@pytest.mark.parametrize(
    "cls,resource", [(CloudPhoneNumber, numbers), (CloudMailbox, mailboxes)]
)
def test_existing_resource_does_not_create(monkeypatch, cls, resource):
    data = SimpleNamespace(
        phone_number="+10000000000",
        device_binding=None,
        state="active",
        checkout_url=None,
        address="a@example.com",
        status="active",
    )
    retrieve = AsyncMock(return_value=SimpleNamespace(data=data))
    create = AsyncMock()
    monkeypatch.setattr(resource, "retrieve", retrieve)
    monkeypatch.setattr(resource, "create", create)

    async def run():
        return await cls("existing", billing="included", plan="rent")

    asyncio.run(run())
    create.assert_not_awaited()
    retrieve.assert_awaited_once()
