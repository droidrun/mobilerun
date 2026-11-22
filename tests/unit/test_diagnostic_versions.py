def test_portal_version_parses_numeric_dev_suffix():
    from mobilerun.diagnostics.versions import parse_portal_version

    assert parse_portal_version("v0.7.26-dev") == (0, 7, 26)
