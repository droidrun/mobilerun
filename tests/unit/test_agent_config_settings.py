"""Shared credential locations remain stable without the add-on."""

from mobilerun.config import settings


def test_existing_credential_locations_stay_unchanged():
    snapshot = settings.get_settings()
    assert snapshot.mobilerun_cloud_credential_path == (
        snapshot.MOBILERUN_USER_CONFIG_DIR / "credentials" / "mobilerun-cloud.json"
    )
