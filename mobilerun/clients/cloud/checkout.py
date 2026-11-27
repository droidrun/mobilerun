"""Explicitly opted-in checkout presentation handlers."""

import os  # Display variables indicate whether this host can open a browser.
import sys
import webbrowser
from typing import Literal

from mobilerun.clients.cloud.payments import CheckoutCallback
from mobilerun.models.checkout import Checkout

CheckoutMode = Literal["error", "print", "browser"]


def print_checkout(checkout: Checkout) -> None:
    expiry = f" (expires {checkout.expires_at})" if checkout.expires_at else ""
    print(f"Complete payment: {checkout.url}{expiry}")


def browser_checkout(checkout: Checkout) -> None:
    print_checkout(checkout)
    if (
        sys.platform != "linux"
        or os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
    ):
        try:
            webbrowser.open(checkout.url)
        except (OSError, webbrowser.Error):
            pass


def checkout_callback(
    mode: CheckoutMode, callback: CheckoutCallback | None
) -> CheckoutCallback | None:
    if mode not in ("error", "print", "browser"):
        raise ValueError(f"Unknown checkout mode: {mode}")
    if mode != "error" and callback is not None:
        raise ValueError("checkout and on_checkout cannot both be set")
    return (
        callback
        if mode == "error"
        else print_checkout if mode == "print" else browser_checkout
    )
