import mobilerun
from mobilerun.clients.portal.setup import portal_version_kwargs


def test_portal_setup_gets_mobilerun_version_when_supported():
    def new_setup(device, debug=False, version=None):
        return None

    def old_setup(device, debug=False):
        return None

    assert portal_version_kwargs(new_setup) == {"version": mobilerun.__version__}
    assert portal_version_kwargs(old_setup) == {}


def test_portal_setup_can_ask_for_an_in_place_update():
    def new_setup(device, debug=False, *, version=None, uninstall=True):
        return None

    def old_setup(device, debug=False):
        return None

    assert portal_version_kwargs(new_setup, in_place=True) == {
        "version": mobilerun.__version__,
        "uninstall": False,
    }
    assert portal_version_kwargs(new_setup) == {"version": mobilerun.__version__}
    assert portal_version_kwargs(old_setup, in_place=True) == {}


def test_sdk_version_check_still_flags_an_rc_after_its_release(monkeypatch):
    import asyncio

    from mobilerun.diagnostics import checks as doctor
    from mobilerun.diagnostics import versions as sdk_releases

    class Response:
        status_code = 200

        def json(self):
            return {"tag_name": "v0.6.22"}

    monkeypatch.setattr(doctor, "__version__", "0.6.22rc1")
    monkeypatch.setattr(sdk_releases.requests, "get", lambda url, **kwargs: Response())

    result = asyncio.run(doctor.check_sdk_version(debug=False))

    assert result.status == doctor.Status.WARN
