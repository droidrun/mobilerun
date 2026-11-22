import pytest

from mobilerun.sandbox import App, SandboxNotImplemented


def test_sandbox_models_export_registration_types():
    from mobilerun.sandbox import Task, routine
    from mobilerun.sandbox.models import Task as AppTask
    from mobilerun.sandbox.models import routine as app_routine

    assert AppTask is Task
    assert app_routine is routine


def _task():
    app = App("test")

    class Input:
        pass

    @app.task()
    def work(input: Input):
        return input

    return work


def test_task_decorator_is_import_safe_and_methods_are_mocks():
    task = _task()
    for fn in (
        lambda: task.local({}),
        lambda: task.deploy(),
        lambda: task.remote({}),
        lambda: task.map([{}]),
    ):
        with pytest.raises(
            SandboxNotImplemented, match="waiting for the sandbox-api release"
        ):
            fn()


def test_task_registration_does_not_call_routines_script(monkeypatch):
    import mobilerun.sandbox.models as task_module

    monkeypatch.setattr(
        task_module,
        "script",
        lambda _: pytest.fail("script decorator called"),
        raising=False,
    )
    _task()


def test_sandbox_client_entry_points_are_mocks():
    from mobilerun.sandbox import client as sandbox

    client = sandbox.SandboxClient()
    calls = [
        lambda: client.create_script(name="x", source=""),
        lambda: client.get_script("id"),
        lambda: client.list_scripts(),
        lambda: client.get_source("id"),
        lambda: client.delete_script("id"),
        lambda: client.wait_until_inspected("id", timeout=1, interval=1),
        lambda: client.start_run("id"),
        lambda: sandbox.create_script(name="x", source=""),
        lambda: sandbox.get_script("id"),
        lambda: sandbox.list_scripts(),
        lambda: sandbox.delete_script("id"),
        lambda: sandbox.wait_until_inspected("id", timeout=1, interval=1),
    ]
    for call in calls:
        with pytest.raises(SandboxNotImplemented):
            call()
