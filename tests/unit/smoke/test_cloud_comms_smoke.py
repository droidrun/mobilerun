"""Opt-in real communications checks; paid resources always cleaned up."""

import asyncio
import secrets
from datetime import datetime, timezone

import pytest
from mobilerun_sdk import APIStatusError

from mobilerun.clients.cloud import mailboxes, numbers
from mobilerun.core.mailbox import CloudMailbox as Mailbox
from mobilerun.core.phone_number import CloudPhoneNumber as PhoneNumber

pytestmark = pytest.mark.smoke


def run_check(coro):
    try:
        return asyncio.run(coro)
    except APIStatusError as exc:
        pytest.fail(
            f"Communications API returned HTTP {exc.status_code}; response omitted"
        )


def test_free_lists_countries_and_capacity(real_env):
    async def exercise():
        await numbers.list_numbers(billing="no_commerce")
        countries = await numbers.countries(billing="no_commerce")
        assert countries.data.items
        await numbers.capacity(countries.data.items[0].country, billing="no_commerce")
        await mailboxes.list_mailboxes(billing="no_commerce")
        await mailboxes.capacity(billing="no_commerce")

    run_check(exercise())


def test_paid_number_create_release_and_sms(real_env, record_property):
    async def exercise():
        countries = await numbers.countries(billing="included")
        country = next(
            (item.country for item in countries.data.items if item.in_stock), None
        )
        if not country:
            pytest.skip("No phone numbers in stock")
        number = await PhoneNumber(
            country=country, billing="paid", label="mobilerun-comms-smoke"
        )
        record_property("created_number", number.id)
        try:
            # Poll for provisioning without creating any further resources.
            info = None
            for _ in range(30):
                info = (await numbers.retrieve(number.id, billing="included")).data
                if info.state != "provisioning":
                    break
                await asyncio.sleep(2)
            if info.state != "active" or not info.actions.send or not info.phone_number:
                record_property(
                    "sms_flow", "unavailable: number not active/send-enabled"
                )
                return
            after = datetime.now(timezone.utc)
            code = str(secrets.randbelow(900000) + 100000)
            await number.send(info.phone_number, "Mobilerun test verification: " + code)
            received = await number.wait_for_code(timeout=60, after=after)
            if received != code:
                pytest.fail("SMS verification did not match; values omitted")
            record_property("sms_flow", "verified")
        finally:
            await number.release()
            remaining = next(
                (
                    item
                    for item in await numbers.list_numbers(billing="included")
                    if item.id == number.id
                ),
                None,
            )
            assert (
                remaining is None
                or remaining.cancel_at_period_end
                or remaining.state in ("expired", "failed")
            )
            record_property("released_number", number.id)

    run_check(exercise())


def test_paid_mailbox_create_release(real_env, record_property):
    async def exercise():
        mailbox = await Mailbox(billing="paid", label="mobilerun-comms-smoke")
        record_property("created_mailbox", mailbox.id)
        try:
            await mailboxes.retrieve(mailbox.id, billing="included")
            # Receiving real email requires an external sender, which this SDK lacks.
            record_property(
                "email_flow", "not run: no authorized outbound email transport"
            )
        finally:
            await mailbox.release()
            remaining = next(
                (
                    item
                    for item in await mailboxes.list_mailboxes(billing="included")
                    if item.id == mailbox.id
                ),
                None,
            )
            assert (
                remaining is None
                or remaining.cancel_at_period_end
                or remaining.status in ("archived", "billing_error")
            )
            record_property("released_mailbox", mailbox.id)

    run_check(exercise())
