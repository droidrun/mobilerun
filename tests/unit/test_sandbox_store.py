"""Sandbox receipt storage keeps scoped ownership evidence safe."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from mobilerun.sandbox import deployment as deploy
from mobilerun.sandbox.errors import SandboxDeploymentConflict, SandboxDeploymentError
from mobilerun.sandbox.models import OwnershipReceipt
from mobilerun.sandbox.settings import SandboxSettings


def scope(origin="https://example.test/v1", key="secret"):
    return deploy.Scope(origin, hashlib.sha256(key.encode()).hexdigest())


def receipt(s, name="../unsafe"):
    return OwnershipReceipt(
        s.origin, s.account_id, name, None, None, None, (), None, ()
    )


def store(tmp_path, s=None, name="../unsafe"):
    return deploy.ReceiptStore(
        s or scope(), name, SandboxSettings(MOBILERUN_SANDBOX_DIRECTORY=str(tmp_path))
    )


def test_scope_uses_cloud_client_key_resolver_and_no_secret(monkeypatch):
    monkeypatch.setattr(deploy, "resolve_cloud_api_key", lambda: "sensitive")
    monkeypatch.setattr(
        deploy,
        "get_settings",
        lambda: SimpleNamespace(MOBILERUN_CLOUD_BASE_URL="https://a"),
    )
    s = deploy.current_scope()
    assert s.origin == "https://a"
    assert s.account_id == hashlib.sha256(b"sensitive").hexdigest()
    assert "sensitive" not in repr(s)
    monkeypatch.setattr(deploy, "resolve_cloud_api_key", lambda: None)
    with pytest.raises(SandboxDeploymentError, match="credentials"):
        deploy.current_scope()


@pytest.mark.parametrize("override", [None, "", "relative/store", "~/cloud-store"])
def test_path_settings_default_and_overrides(tmp_path, monkeypatch, override):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "mobilerun.sandbox.settings.get_settings",
        lambda: SimpleNamespace(MOBILERUN_USER_CONFIG_DIR=tmp_path / "config"),
    )
    settings = SandboxSettings(MOBILERUN_SANDBOX_DIRECTORY=override)
    st = deploy.ReceiptStore(scope(), "../unsafe", settings)
    root = {
        None: tmp_path / "config/sandbox",
        "": tmp_path / "config/sandbox",
        "relative/store": tmp_path / "relative/store",
        "~/cloud-store": tmp_path / "cloud-store",
    }[override]
    assert (
        st.directory
        == root
        / hashlib.sha256(
            (scope().origin + "\0" + scope().account_id).encode()
        ).hexdigest()
        / hashlib.sha256(b"../unsafe").hexdigest()
    )
    assert st.read() is None
    assert not root.exists()


def test_scope_origin_account_and_names_are_isolated(tmp_path):
    assert (
        len(
            {
                store(tmp_path, scope(), "n").directory,
                store(tmp_path, scope(key="other"), "n").directory,
                store(tmp_path, scope(origin="https://other"), "n").directory,
                store(tmp_path, scope(), "../n").directory,
            }
        )
        == 4
    )


def test_atomic_write_preserves_old_receipt_on_replace_failure(tmp_path, monkeypatch):
    st = store(tmp_path)
    r = receipt(scope())
    st.write(r)

    def crash(*args):
        raise OSError("secret detail")

    monkeypatch.setattr(deploy.os, "replace", crash)
    with pytest.raises(SandboxDeploymentError) as exc:
        st.write(replace(r, function_id="new"))
    assert "secret detail" not in str(exc.value)
    assert st.read() == r
    assert list(st.directory.glob(".sandbox-*"))


@pytest.mark.parametrize("bad", ["not json", "{}", '{"origin": 3}', "[]"])
def test_malformed_receipts_fail_closed(tmp_path, bad):
    st = store(tmp_path)
    st.directory.mkdir(parents=True)
    st.receipt_path.write_text(bad)
    with pytest.raises(SandboxDeploymentError):
        st.read()


def test_scope_and_name_mismatch_cannot_be_adopted(tmp_path):
    st = store(tmp_path)
    st.directory.mkdir(parents=True)
    st.receipt_path.write_text(json.dumps(receipt(scope(key="different")).to_dict()))
    with pytest.raises(SandboxDeploymentConflict):
        st.read()


def test_identity_lock_conflicts_and_releases(tmp_path):
    st = store(tmp_path)
    with st.lock():
        with pytest.raises(SandboxDeploymentConflict):
            with st.lock():
                pass
    with st.lock():
        pass


def test_stage_intent_and_ids_are_persisted_and_stable(tmp_path):
    s = scope()
    st = store(tmp_path)
    r = receipt(s)
    key = deploy.stage_key(s, r.name, "digest", "action:create:0")
    assert key == deploy.stage_key(s, r.name, "digest", "action:create:0")
    assert key != deploy.stage_key(s, r.name, "digest", "action:create:1")
    r = st.record_stage(r, "action:create:0", key)
    assert st.read() == r
    assert key in r.stages[-1] and "action:create:0" in r.stages[-1]
    r = st.record_ids(r, function_id="f", action_ids=("a",))
    assert st.read() == r
    assert r.function_id == "f" and r.action_ids == ("a",)


def test_yaml_quotes_on_and_only_removes_with_confirmation(tmp_path):
    st = store(tmp_path)
    st.write_yaml({"on": {"custom": {}}, "version": 1})
    assert '"on":' in st.yaml_path.read_text()
    assert yaml.safe_load(st.yaml_path.read_text())["on"] == {"custom": {}}
    with pytest.raises(SandboxDeploymentError):
        st.remove_yaml(cleanup_confirmed=False)
    assert st.yaml_path.exists()
    st.remove_yaml(cleanup_confirmed=True)
    assert not st.yaml_path.exists()


def test_windows_lock_backend_is_selected_and_released(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []
    fake = SimpleNamespace(
        LK_NBLCK=1,
        LK_UNLCK=2,
        locking=lambda fd, mode, size: calls.append((mode, size)),
    )
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(deploy, "_windows_platform", lambda: True, raising=False)
    with store(tmp_path).lock():
        assert calls == [(1, 1)]
    assert calls == [(1, 1), (2, 1)]
