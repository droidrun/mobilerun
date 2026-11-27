"""Error hierarchy, safe diagnostics and typed cloud failures."""

import ast
import asyncio
import importlib
import logging
import traceback
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import APIStatusError

from mobilerun.clients.cloud import mailboxes, numbers
from mobilerun.clients.portal.state_fetch import fetch_state_with_retry
from mobilerun.core.extensions import Extension, action
from mobilerun.errors import (
    BillingError,
    CloudConflict,
    CloudServiceUnavailable,
    ExpectationInconclusive,
    ExtensionActionError,
    StateFetchFailed,
)
from mobilerun.errors.base import CloudError, DeviceError, MobilerunError, is_value_safe
from mobilerun.models.billing import Billing
from tests.support.paths import REPO_ROOT


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("mobilerun.errors.credentials", "CredentialNotFoundError"),
        ("mobilerun.errors.device", "DeviceAuthError"),
        ("mobilerun.errors.device", "DeviceNotFound"),
        (
            "mobilerun.errors.device",
            "DeviceOperationNotSupported",
        ),
        ("mobilerun.errors.operators", "ExpectationFailed"),
        ("mobilerun.errors.cloud", "InvalidAPIKeyError"),
        (
            "mobilerun.errors.credentials",
            "LocalCredentialsNotAllowed",
        ),
        ("mobilerun.errors.operators", "OperatorNotAvailable"),
        ("mobilerun.errors.operators", "OperatorNotSupported"),
        ("mobilerun.errors.device", "PortalApkNotFound"),
        ("mobilerun.errors.device", "PortalInstallFailed"),
        ("mobilerun.errors.device", "UnsupportedStateCapability"),
        ("mobilerun.errors.device", "StateFetchFailed"),
        ("mobilerun.macro.errors", "MacroMismatch"),
        ("mobilerun.config.loader", "OutdatedConfigError"),
    ],
)
def test_errors_share_root(module, name):
    assert issubclass(getattr(importlib.import_module(module), name), MobilerunError)


def test_new_error_types_keep_builtin_compatibility():
    from mobilerun.errors import CredentialFormatError, DeviceNotReady, PlaceholderError

    assert issubclass(CredentialFormatError, ValueError)
    assert issubclass(DeviceNotReady, RuntimeError)
    assert issubclass(PlaceholderError, ValueError)


def test_invalid_values_raise_typed_errors():
    from mobilerun.core.credentials.base import normalize
    from mobilerun.core.state.view import UIStateView
    from mobilerun.errors import CredentialFormatError, DeviceNotReady

    with pytest.raises(CredentialFormatError):
        normalize({"app": 123})
    with pytest.raises(DeviceNotReady):
        _ = UIStateView(None).current


def test_error_classes_discovered_by_ast_share_root():
    root = REPO_ROOT / "mobilerun"
    named_errors = {
        "DeviceNotFound",
        "DeviceNotReady",
        "DeviceOperationNotSupported",
        "ExpectationFailed",
        "ExpectationInconclusive",
        "LocalCredentialsNotAllowed",
        "OperatorNotAvailable",
        "OperatorNotSupported",
        "PortalApkNotFound",
        "PortalInstallFailed",
        "StateFetchFailed",
        "UnsupportedStateCapability",
        "MacroMismatch",
        "SandboxNotImplemented",
    }
    for path in root.rglob("*.py"):
        if path.name == "__init__.py" or "clients/cloud" in str(path):
            continue
        module = ".".join(path.relative_to(root.parent).with_suffix("").parts)
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name.startswith("Cloud") or node.name == "MobilerunError":
                continue
            if node.name.endswith("Error") or node.name in named_errors:
                cls = getattr(importlib.import_module(module), node.name)
                if issubclass(cls, BaseException):
                    assert issubclass(cls, MobilerunError), f"{module}.{node.name}"


def test_inconclusive_verdict_keeps_sensitive_details_out_of_message():
    task = SimpleNamespace(answer="secret-model-output")
    error = ExpectationInconclusive("secret-instruction", task)
    assert "secret" not in str(error)
    assert error.condition == "secret-instruction"
    assert error.task is task
    assert isinstance(error, DeviceError)


def test_portal_error_does_not_expose_device_state(caplog):
    async def fetch():
        return {"error": "secret-error", "phone_state": "secret-state"}

    with pytest.raises(StateFetchFailed) as caught:
        asyncio.run(fetch_state_with_retry(fetch, max_retries=1, retry_delays=[0]))
    assert "secret" not in str(caught.value)
    assert "secret" not in caplog.text


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_sensitive_direct_free_text_cloud_error_is_wrapped(level, caplog):
    caplog.set_level(level, logger="mobilerun")
    canary = "PRIVATE-CANARY"

    class Vault(Extension):
        @action(sensitive_parameters=frozenset({"password"}))
        async def login(self, password: str) -> None:
            """Reject unsafe legacy cloud messages."""
            raise CloudServiceUnavailable(password)

    with pytest.raises(ExtensionActionError) as caught:
        asyncio.run(Vault().login(canary))
    assert caught.value.cause_type == "CloudServiceUnavailable"
    assert caught.value.__cause__ is None
    assert canary not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
@pytest.mark.parametrize("sensitive", [False, True])
def test_real_number_messages_wrapper_drops_unsafe_sdk_chain(level, sensitive, caplog):
    caplog.set_level(level, logger="mobilerun")
    canary = "SDK-PRIVATE-CANARY"
    request = httpx.Request("GET", "https://example.invalid/numbers")
    response = httpx.Response(
        503, request=request, headers={"x-request-id": "support-id"}
    )
    original = APIStatusError(canary, response=response, body={"detail": canary})
    sdk = SimpleNamespace(
        numbers=SimpleNamespace(
            messages=SimpleNamespace(list=AsyncMock(side_effect=original))
        )
    )

    class Inbox(Extension):
        @action(
            sensitive_parameters=frozenset({"password"}) if sensitive else frozenset()
        )
        async def read(self, password: str) -> None:
            """Read through the real SDK translation boundary."""
            await numbers.messages("number-id", billing=Billing.NO_COMMERCE, client=sdk)

    with pytest.raises(CloudServiceUnavailable) as caught:
        asyncio.run(Inbox().read(canary))
    error = caught.value
    assert error.value_safe
    assert error.status == 503
    assert error.operation == "List SMS messages"
    assert error.request_id == "support-id"
    assert error.__cause__ is None
    assert error.__context__ is None
    assert canary not in "".join(traceback.format_exception(error))
    sdk.numbers.messages.list.assert_awaited_once()


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_safe_chain_preserves_identity_metadata_and_only_safe_causes(
    explicit, level, caplog
):
    caplog.set_level(level, logger="mobilerun")
    canary = "CHAIN-PRIVATE-CANARY"
    unsafe = CloudError(canary)
    safe_cause = CloudError(operation="List SMS messages", status=503)
    safe_cause.__cause__ = unsafe
    top = CloudServiceUnavailable(status=503, detail={"raw": canary})

    class Resource(Extension):
        @action(sensitive_parameters=frozenset({"password"}))
        async def run(self, password: str) -> None:
            """Retain the typed error, not backend contexts."""
            try:
                raise safe_cause
            except CloudError:
                if explicit:
                    raise top from safe_cause
                raise top  # noqa: B904 -- exercise implicit SDK-style context.

    with pytest.raises(CloudServiceUnavailable) as caught:
        asyncio.run(Resource().run(canary))
    assert caught.value is top
    assert top.detail == {"raw": canary}
    assert top.__context__ is None
    assert top.__cause__ is (safe_cause if explicit else None)
    if explicit:
        assert safe_cause.__cause__ is None
        assert safe_cause.__context__ is None
    assert canary not in "".join(traceback.format_exception(top))


def test_safe_cause_cycles_are_removed():
    top, cause = CloudError(), CloudError()
    top.__cause__, cause.__cause__ = cause, top

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Avoid retaining cyclic safe exception chains."""
            raise top

    with pytest.raises(CloudError) as caught:
        asyncio.run(Resource().run())
    assert caught.value is top
    assert top.__cause__ is cause
    assert cause.__cause__ is None


def test_typed_cloud_fields_do_not_render_arbitrary_labels_or_status():
    canary = "FIELD-PRIVATE-CANARY"
    error = CloudError(
        operation=canary, status=canary, detail=canary, request_id="invalid/value"
    )
    assert error.value_safe
    assert error.status == error.detail == error.operation == canary
    assert canary not in str(error)
    assert "invalid/value" not in str(error)
    assert not CloudError(canary).value_safe


def test_unknown_cloud_error_code_is_rejected():
    with pytest.raises(TypeError, match="Unknown cloud error code"):
        CloudError(code="unknown")


@pytest.mark.parametrize(
    "operation",
    [
        "Retrieve cloud phone location",
        "Retrieve cloud phone language",
        "Retrieve cloud phone timezone",
        "Release proxy",
    ],
)
def test_new_cloud_operation_labels_are_retained(operation):
    assert str(CloudError(operation=operation, status=503)).startswith(
        f"{operation} failed (HTTP 503)."
    )


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
@pytest.mark.parametrize("on_cause", [False, True])
def test_safe_errors_drop_notes_on_every_retained_exception(level, on_cause, caplog):
    caplog.set_level(level, logger="mobilerun")
    canary = "NOTE-PRIVATE-CANARY"
    top, cause = CloudServiceUnavailable(status=503), CloudError(status=503)
    top.__cause__ = cause

    class Resource(Extension):
        @action(sensitive_parameters=frozenset({"password"}))
        async def run(self, password: str) -> None:
            """Do not propagate provider notes as safe diagnostics."""
            (cause if on_cause else top).add_note(password)
            raise top

    with pytest.raises(CloudServiceUnavailable) as caught:
        asyncio.run(Resource().run(canary))
    assert caught.value is top
    assert top.__cause__ is cause
    assert not getattr(top, "__notes__", [])
    assert not getattr(cause, "__notes__", [])
    assert canary not in "".join(traceback.format_exception(top))


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
@pytest.mark.parametrize("sensitive", [False, True])
@pytest.mark.parametrize("bypass", ["str", "repr", "init", "args", "opted_str"])
def test_cloud_rendering_bypasses_are_wrapped(level, sensitive, bypass, caplog):
    caplog.set_level(level, logger="mobilerun")
    canary = "RENDER-PRIVATE-CANARY"

    class CustomString(CloudServiceUnavailable):
        def __str__(self):
            return self.detail

    class CustomRepr(CloudServiceUnavailable):
        def __repr__(self):
            return self.detail

    class CustomInit(CloudServiceUnavailable):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class OptedString(CustomString):
        safe_rendering = True

        def __str__(self):
            return self.detail

    error_type = {
        "str": CustomString,
        "repr": CustomRepr,
        "init": CustomInit,
        "args": CloudServiceUnavailable,
        "opted_str": OptedString,
    }[bypass]
    original = error_type(status=503, detail=canary)

    class Resource(Extension):
        @action(
            sensitive_parameters=frozenset({"password"}) if sensitive else frozenset()
        )
        async def run(self, password: str) -> None:
            """Reject mutable or unaudited error rendering."""
            if bypass == "args":
                original.args = (password,)
            raise original

    with pytest.raises(ExtensionActionError) as caught:
        asyncio.run(Resource().run(canary))
    assert caught.value.__cause__ is None
    assert canary not in "".join(traceback.format_exception(caught.value))
    if bypass == "args":
        assert canary not in str(original)


def test_private_cloud_message_is_write_once():
    original = CloudServiceUnavailable(status=503)
    message = str(original)
    original.args = ("mutated",)
    assert str(original) == message
    with pytest.raises(AttributeError):
        original._CloudError__safe_message = "mutated"
    with pytest.raises(AttributeError):
        del original._CloudError__safe_message


@pytest.mark.parametrize(
    "contract", ["stock", "inherited", "explicit", "payment", "purchase"]
)
def test_audited_or_explicit_safe_rendering_retains_error_identity(contract):
    from mobilerun.errors import PaymentRequired, PurchaseInProgress
    from mobilerun.models.checkout import Checkout

    class Inherited(CloudServiceUnavailable):
        pass

    class Explicit(CloudServiceUnavailable):
        safe_rendering = True

        def __init__(self):
            super().__init__(status=503)

        def __str__(self):
            return super().__str__()

    async def resume(timeout):
        return None

    checkout = Checkout("https://example.invalid/checkout", None, "number", "id")
    original = {
        "stock": lambda: CloudServiceUnavailable(status=503),
        "inherited": lambda: Inherited(status=503),
        "explicit": Explicit,
        "payment": lambda: PaymentRequired(checkout, resume),
        "purchase": lambda: PurchaseInProgress("id", checkout),
    }[contract]()

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Keep audited renderers and explicit safe-rendering opt-ins."""
            raise original

    with pytest.raises(type(original)) as caught:
        asyncio.run(Resource().run())
    assert caught.value is original


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
@pytest.mark.parametrize("bypass", ["str", "args"])
def test_retained_cloud_causes_are_revalidated(level, bypass, caplog):
    caplog.set_level(level, logger="mobilerun")
    canary = "CAUSE-RENDER-PRIVATE-CANARY"

    class Custom(CloudServiceUnavailable):
        def __str__(self):
            return self.detail

    cause = Custom(status=503, detail=canary) if bypass == "str" else CloudError()
    if bypass == "args":
        cause.args = (canary,)
    top = CloudServiceUnavailable(status=503)
    top.__cause__ = cause

    class Resource(Extension):
        @action(sensitive_parameters=frozenset({"password"}))
        async def run(self, password: str) -> None:
            """Drop a cause whose rendering contract no longer holds."""
            raise top

    with pytest.raises(CloudServiceUnavailable) as caught:
        asyncio.run(Resource().run(canary))
    assert caught.value is top
    assert top.__cause__ is None
    assert canary not in "".join(traceback.format_exception(top))


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
@pytest.mark.parametrize("as_cause", [False, True])
def test_computed_hint_is_never_trusted(level, as_cause, caplog):
    caplog.set_level(level, logger="mobilerun")
    canary = "HINT-PRIVATE-CANARY"

    class ComputedHint(CloudServiceUnavailable):
        @property
        def _hint(self):
            return str(self.detail)

    original = ComputedHint(status=503, detail=canary)
    top = CloudServiceUnavailable(status=503) if as_cause else original
    if as_cause:
        top.__cause__ = original

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Apply the allowlist to top-level errors and retained causes."""
            raise top

    if as_cause:
        with pytest.raises(CloudServiceUnavailable) as caught:
            asyncio.run(Resource().run())
        assert caught.value is top
    else:
        with pytest.raises(ExtensionActionError) as caught:
            asyncio.run(Resource().run())
    assert caught.value.__cause__ is None
    assert canary not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
@pytest.mark.parametrize(
    "addition",
    [
        "method",
        "property",
        "classmethod",
        "staticmethod",
        "descriptor",
        "getattribute",
        "getattr",
        "args",
        "mutable_data",
        "nested_behavior",
    ],
)
def test_extra_subclass_behavior_is_untrusted(level, addition, caplog):
    caplog.set_level(level, logger="mobilerun")

    def extra(self):
        return "not-rendering-related"

    class Descriptor:
        def __get__(self, instance, owner):
            return "not-rendering-related"

    attrs = {
        "method": {"extra": extra},
        "property": {"extra": property(extra)},
        "classmethod": {"extra": classmethod(extra)},
        "staticmethod": {"extra": staticmethod(extra)},
        "descriptor": {"extra": Descriptor()},
        "mutable_data": {"extra": ["not-a-constant"]},
        "nested_behavior": {"extra": ("constant", (extra,))},
        "getattribute": {
            "__getattribute__": lambda self, name: object.__getattribute__(self, name)
        },
        "getattr": {"__getattr__": lambda self, name: None},
        "args": {"args": property(lambda self: (CloudError.__str__(self),))},
    }[addition]
    custom = type("ExtraBehavior", (CloudServiceUnavailable,), attrs)
    original = custom(status=503)

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Unknown subclass behavior is not audited by inheritance."""
            raise original

    with pytest.raises(ExtensionActionError) as caught:
        asyncio.run(Resource().run())
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "attrs",
    [
        {"retryable": True},
        {"extra": "static-constant"},
        {"metadata": ("retry", (1, 2.5, False, None), frozenset({"safe"}))},
        {"__slots__": ()},
        {"__annotations__": {"retryable": bool}, "retryable": True},
    ],
)
def test_plain_constant_subclasses_retain_identity(attrs):
    custom = type("ConstantData", (CloudServiceUnavailable,), attrs)
    original = custom(status=503)

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Constant class metadata does not add rendering behavior."""
            raise original

    with pytest.raises(custom) as caught:
        asyncio.run(Resource().run())
    assert caught.value is original


@pytest.mark.parametrize("parent_behavior", [False, True])
def test_explicit_opt_in_only_vouches_for_its_own_class(parent_behavior):
    attrs = {"retryable": True}
    if parent_behavior:
        attrs["retry"] = lambda self: None
    parent = type("Parent", (CloudServiceUnavailable,), attrs)

    class OptedIn(parent):
        safe_rendering = True

        def retry(self):
            return None

    assert is_value_safe(OptedIn(status=503)) is (not parent_behavior)


def test_static_classvar_hint_is_allowed_and_not_read_from_instance():
    from typing import ClassVar

    class StaticHint(CloudServiceUnavailable):
        _hint: ClassVar[str] = "Wait for the service to recover."

    original = StaticHint(status=503)
    assert "Wait for the service to recover." in str(original)

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Static allowlisted ClassVar data does not alter behavior."""
            raise original

    with pytest.raises(StaticHint) as caught:
        asyncio.run(Resource().run())
    assert caught.value is original


def test_mro_does_not_hide_behavior_after_the_cloud_base():
    class Behavior:
        def unrelated(self):
            return None

    class Mixed(CloudServiceUnavailable, Behavior):
        pass

    original = Mixed(status=503)

    class Resource(Extension):
        @action
        async def run(self) -> None:
            """Inspect every unaudited MRO entry, including late mixins."""
            raise original

    with pytest.raises(ExtensionActionError):
        asyncio.run(Resource().run())


@pytest.mark.parametrize(
    "module,resource,plural",
    [(numbers, "numbers", "phone numbers"), (mailboxes, "mailboxes", "mailboxes")],
)
@pytest.mark.parametrize("code", ["included_capacity_unavailable", "other_conflict"])
def test_create_capacity_conflict(module, resource, plural, code):
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(409, request=request)
    error = APIStatusError("conflict", response=response, body={"code": code})
    sdk = SimpleNamespace(
        **{resource: SimpleNamespace(create=AsyncMock(side_effect=error))}
    )
    with pytest.raises(
        BillingError if code == "included_capacity_unavailable" else CloudConflict
    ) as caught:
        asyncio.run(
            module.create(
                billing=Billing.PAID,
                client=sdk,
                **({"country": "US"} if resource == "numbers" else {}),
            )
        )
    assert is_value_safe(caught.value)
    if code == "included_capacity_unavailable":
        assert f"No included {plural} left" in str(caught.value)
        if resource == "numbers":
            assert "No included phone numbers left for US." in str(caught.value)
        assert 'plan="auto"' in str(caught.value)
        assert 'plan="rent"' in str(caught.value)
        assert 'billing="paid"' in str(caught.value)
    else:
        assert "Wait for the current operation" in str(caught.value)


@pytest.mark.parametrize(
    "country,expected",
    [
        ("us", "US"),
        ("gB", "GB"),
        ("US", "US"),
        ("PRIVATE-CANARY", None),
        ("UＳ", None),
        ("ß", None),
        (" US ", None),
        (123, None),
        (None, None),
    ],
)
def test_capacity_country_is_validated_before_rendering(country, expected):
    from mobilerun.errors import IncludedCapacityUnavailable

    error = IncludedCapacityUnavailable("number", country=country)
    assert is_value_safe(error)
    assert error.country == expected
    assert (f"for {expected}." if expected else "for the requested country.") in str(
        error
    )
    assert "PRIVATE-CANARY" not in str(error)


def test_capacity_country_from_client_never_renders_arbitrary_text():
    request = httpx.Request("POST", "https://example.test")
    error = APIStatusError(
        "PRIVATE-BACKEND",
        response=httpx.Response(409, request=request),
        body={"code": "included_capacity_unavailable"},
    )
    sdk = SimpleNamespace(numbers=SimpleNamespace(create=AsyncMock(side_effect=error)))
    with pytest.raises(BillingError) as caught:
        asyncio.run(
            numbers.create(
                billing=Billing.INCLUDED, client=sdk, country="PRIVATE-COUNTRY"
            )
        )
    assert is_value_safe(caught.value)
    assert "PRIVATE" not in str(caught.value)


def test_capacity_error_rejects_unknown_resource_without_echoing_it():
    from mobilerun.errors import IncludedCapacityUnavailable

    with pytest.raises(ValueError) as caught:
        IncludedCapacityUnavailable("PRIVATE-RESOURCE")
    assert "PRIVATE" not in str(caught.value)
