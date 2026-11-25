"""Public error groups shared by Mobilerun subsystems."""

import re
from types import GetSetDescriptorType
from typing import ClassVar, Generic, Literal

CloudErrorCode = Literal[
    "request_failed",
    "webhook_duplicate_url",
    "webhook_registration_in_progress",
    "webhook_replay_deleted",
    "missing_token",
    "auth_failed",
    "invalid_billing",
    "billing_override",
    "payment_failed",
    "checkout_missing",
    "task_timeout",
    "credentials_required",
    "purchase_in_progress",
    "payment_required",
    "number_unready",
    "proxy_disconnect",
    "included_number_capacity",
    "included_mailbox_capacity",
    "unsupported_number_country",
    "device_type_unavailable",
]

_OPERATIONS = frozenset(
    {
        "Cloud request",
        "create",
        "Bind phone number",
        "Buy Connect proxy",
        "Cloud phone request",
        "Connect phone proxy",
        "Contact authentication service",
        "Create cloud phone",
        "Create mailbox",
        "Create number",
        "Create vault field",
        "Create vault package",
        "Delete device file",
        "Delete file",
        "Delete vault credential",
        "Delete workspace app",
        "Delete webhook",
        "Get webhook",
        "Ping webhook",
        "List webhooks",
        "Register webhook",
        "Disconnect phone proxy",
        "Download device file",
        "Download file",
        "Exchange device code",
        "Install phone app",
        "List SMS messages",
        "List cloud apps",
        "List cloud phones",
        "List device files",
        "List files",
        "List mailboxes",
        "List number countries",
        "List numbers",
        "List phone app installs",
        "List proxies",
        "List vault credentials",
        "List vault package",
        "Poll cloud task",
        "Prepare cloud app upload",
        "Prepare file upload",
        "Reboot cloud phone",
        "Release mailbox",
        "Release number",
        "Release proxy",
        "Rename cloud phone",
        "Request device code",
        "Reset cloud phone",
        "Resume cloud phone",
        "Retrieve authentication session",
        "Retrieve cloud app",
        "Retrieve cloud phone",
        "Retrieve cloud phone capabilities",
        "Retrieve cloud phone location",
        "Retrieve cloud phone language",
        "Retrieve cloud phone timezone",
        "Retrieve cloud task screenshots",
        "Retrieve cloud task trajectory",
        "Retrieve mailbox",
        "Retrieve mailbox OTP",
        "Retrieve mailbox capacity",
        "Retrieve number",
        "Retrieve number capacity",
        "Retrieve phone number binding",
        "Retrieve phone proxy status",
        "Retrieve proxy",
        "Run cloud task",
        "Save vault credential",
        "Send SMS",
        "Stop cloud phone",
        "Stop cloud task",
        "Stream cloud task progress",
        "Terminate cloud phone",
        "Unbind phone number",
        "Uninstall phone app",
        "Update vault field",
        "Upload cloud app",
        "Upload device file",
        "Upload file",
        "Validate cloud API key",
        "Wait for cloud phone",
    }
)

_BILLING_OPERATIONS = frozenset(
    {
        "bind_number",
        "buy_proxy",
        "cancel_proxy",
        "capabilities",
        "capacity",
        "cloud_client",
        "connect_device",
        "countries",
        "create",
        "create_field",
        "create_package",
        "delete",
        "delete_app",
        "delete_device_file",
        "delete_file",
        "device_status",
        "disconnect_device",
        "download_device_file",
        "download_file",
        "fetch_session",
        "get_app",
        "get_location",
        "get_language",
        "get_timezone",
        "get_proxy",
        "install",
        "list_apps",
        "get_webhook",
        "ping_webhook",
        "list_webhooks",
        "register_webhook",
        "delete_webhook",
        "list_device_files",
        "list_devices",
        "list_files",
        "list_installs",
        "list_mailboxes",
        "list_numbers",
        "list_package",
        "list_proxies",
        "list_secrets",
        "messages",
        "number",
        "otp",
        "poll_token",
        "progress",
        "push_device_file",
        "reboot",
        "request_code",
        "reset",
        "resume",
        "retrieve",
        "run",
        "run_task",
        "save",
        "screenshots",
        "send",
        "set_name",
        "status",
        "stop",
        "terminate",
        "trajectory",
        "unbind_number",
        "uninstall",
        "update",
        "upload_app",
        "upload_file",
        "validate_api_key",
        "wait_ready",
    }
)


def _reference(value: str | None) -> str | None:
    return (
        value
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value)
        else None
    )


class MobilerunError(Exception):
    # Opt in only when all messages are value-safe, including subclass messages.
    value_safe: bool = False


class ConfigError(MobilerunError):
    pass


class DeviceError(MobilerunError):
    pass


class CloudError(MobilerunError):
    """Typed diagnostics are safe; legacy free-text messages are not trusted."""

    _hint: ClassVar[str] = "Check the operation parameters and retry."
    safe_rendering: ClassVar[bool] = True

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_CloudError__safe_message" and name in self.__dict__:
            raise AttributeError("Cloud error message is immutable")
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name == "_CloudError__safe_message":
            raise AttributeError("Cloud error message is immutable")
        super().__delattr__(name)

    def __str__(self) -> str:
        return self.__safe_message

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__safe_message!r})"

    def __init__(
        self,
        message: str | None = None,
        *,
        operation: str = "Cloud request",
        status: int | str | None = None,
        code: CloudErrorCode = "request_failed",
        request_id: str | None = None,
        detail: object = None,
        timeout: float | None = None,
        resource_id: str | None = None,
        required_billing: Literal["no_commerce", "included", "paid"] | None = None,
        country: object = None,
    ) -> None:
        self.operation, self.status, self.code = operation, status, code
        self.request_id, self.detail = request_id, detail
        self.timeout, self.resource_id = timeout, resource_id
        self.required_billing = required_billing
        normalized_country = (
            country.upper()
            if type(country) is str and re.fullmatch(r"[A-Za-z]{2}", country)
            else None
        )
        self.country = normalized_country
        self.value_safe = message is None
        if message is not None:
            self.__safe_message = message
            super().__init__(message)
            return
        label = operation if operation in _OPERATIONS else "Cloud request"
        http = (
            f"HTTP {status}"
            if type(status) is int and 100 <= status <= 599
            else "transport error"
        )
        request = _reference(request_id)
        reference = f"; request id {request}" if request else ""
        # Read declared data, never invoke a subclass property or lookup hook.
        hint = CloudError._hint
        for base in type.__getattribute__(type(self), "__mro__"):
            fields = type.__getattribute__(base, "__dict__")
            if "_hint" in fields:
                if type(fields["_hint"]) is str:
                    hint = fields["_hint"]
                break
        hints = {
            "webhook_duplicate_url": "A webhook already exists for this URL.",
            "webhook_registration_in_progress": "Webhook registration is still in progress; retry with the same key.",
            "webhook_replay_deleted": "The webhook created with this key has been deleted.",
            "request_failed": hint,
            "missing_token": "Token response missing access_token; run `mobilerun account login` again.",
            "auth_failed": "Run `mobilerun account login` again.",
            "billing_override": "Billing and proxy fields cannot be overridden through extra parameters.",
            "payment_failed": "Cloud resource payment failed.",
            "checkout_missing": "Cloud resource left payment state without a checkout URL.",
            "credentials_required": "No Mobilerun Cloud credentials. Run `mobilerun account login`, set MOBILERUN_API_KEY, or pass api_key=.",
            "purchase_in_progress": "A phone-number purchase is already in progress. Complete checkout or cancel it with await CloudPhoneNumber(error.resource_id).release().",
            "payment_required": "Payment required. Complete checkout using error.checkout.url. Then call await error.wait_until_paid().",
            "number_unready": "Number cannot become ready; check its status in the dashboard.",
            "proxy_disconnect": "The backend could not disconnect the proxy. Check the phone state or contact support; no retry was attempted.",
            "included_number_capacity": f'No included phone numbers left for {normalized_country or "the requested country"}. Use plan="auto" (included first, then rent) or plan="rent" with billing="paid".',
            "included_mailbox_capacity": 'No included mailboxes left. Use plan="auto" (included first, then rent) or plan="rent" with billing="paid".',
            "unsupported_number_country": "The requested country is not available. Choose one from `mobilerun number countries`.",
            "device_type_unavailable": "The requested cloud phone type is unavailable.",
        }
        if code == "invalid_billing":
            label = operation if operation in _BILLING_OPERATIONS else "Cloud request"
            mode = (
                required_billing
                if required_billing in {"no_commerce", "included", "paid"}
                else "included"
            )
            rendered = f"{label}: given billing policy is invalid; requires billing={mode!r}. Pass billing={mode!r} or set MOBILERUN_BILLING={mode}."
        elif code == "task_timeout":
            task = _reference(resource_id)
            duration = (
                f" within {timeout:g} seconds"
                if isinstance(timeout, (int, float)) and 0 <= timeout < float("inf")
                else ""
            )
            rendered = f"Cloud task{' ' + task if task else ''} did not finish{duration}. Check the task dashboard."
        else:
            if code not in hints:
                raise TypeError("Unknown cloud error code")
            rendered = f"{label} failed ({http}{reference}). {hints[code]}"
        self.__safe_message = rendered
        super().__init__(rendered)


# Audited foundations used by stock cloud errors, including multiple inheritance.
_AUDITED_CLOUD_BASES = frozenset(
    (*MobilerunError.__mro__, *TimeoutError.__mro__, *ValueError.__mro__, Generic)
)


def _plain_constant(value: object) -> bool:
    if type(value) in (str, int, float, bool, type(None)):
        return True
    return type(value) in (tuple, frozenset) and all(
        _plain_constant(item) for item in value
    )


def _safe_cloud_class_data(cls: type[CloudError]) -> bool:
    """Each MRO entry must declare its own opt-in or add only plain data."""
    for base in type.__getattribute__(cls, "__mro__"):
        if base in _AUDITED_CLOUD_BASES:
            continue
        fields = type.__getattribute__(base, "__dict__")
        # This must be declared on each class, never inherited as an audit grant.
        if fields.get("safe_rendering") is True:
            continue
        for name, value in fields.items():
            if name == "__annotations__":
                if type(value) is not dict:
                    return False
            elif name == "__type_params__":
                if type(value) is not tuple or value:
                    return False
            elif name in {"__dict__", "__weakref__"}:
                if type(value) is not GetSetDescriptorType:
                    return False
            elif name == "__slots__":
                if type(value) is not tuple or any(
                    type(item) is not str for item in value
                ):
                    return False
            elif not _plain_constant(value):
                return False
    return True


def is_value_safe(error: BaseException) -> bool:
    """Revalidate cloud rendering at each boundary, not just construction."""
    if not isinstance(error, MobilerunError):
        return False
    if not isinstance(error, CloudError):
        return error.value_safe
    # Check declared metadata before any instance lookup can run subclass code.
    if not _safe_cloud_class_data(type(error)):
        return False
    try:
        if not error.value_safe:
            return False
        message = CloudError.__str__(error)
        # Mutable args must not smuggle different text into exception consumers.
        return (
            type(message) is str
            and len(error.args) == 1
            and type(error.args[0]) is str
            and error.args[0] == message
            and str(error) == message
        )
    except Exception:
        return False


class CredentialError(MobilerunError):
    pass


class OperatorError(MobilerunError):
    pass


class AgentError(MobilerunError):
    pass
