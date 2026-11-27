"""One check decides whether a prompt on a given stream can be answered."""

import sys
from types import SimpleNamespace

import pytest

from mobilerun.cli.base import menu
from mobilerun.clients.cloud import billing
from mobilerun.utils.terminal import can_prompt


def _stream(tty):
    return None if tty is None else SimpleNamespace(isatty=lambda: tty)


@pytest.mark.parametrize(
    "stdin,output,expected",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (None, True, False),
        (True, None, False),
    ],
)
def test_prompt_needs_terminal_stdin_and_output(monkeypatch, stdin, output, expected):
    monkeypatch.setattr(sys, "stdin", _stream(stdin))
    assert can_prompt(_stream(output)) is expected


@pytest.mark.parametrize(
    "stdout,stderr,menu_prompts,billing_prompts",
    [(True, False, True, False), (False, True, False, True), (None, True, False, True)],
)
def test_cli_and_billing_checks_differ_only_by_stream(
    monkeypatch, stdout, stderr, menu_prompts, billing_prompts
):
    monkeypatch.setattr(sys, "stdin", _stream(True))
    monkeypatch.setattr(sys, "stdout", _stream(stdout))
    monkeypatch.setattr(sys, "stderr", _stream(stderr))
    results = menu.is_interactive(), billing.billing_prompt_is_interactive()
    monkeypatch.undo()
    assert results == (menu_prompts, billing_prompts)
