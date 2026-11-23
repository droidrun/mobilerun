def test_removed_base_config_legacy_name():
    from mobilerun.config import user_config

    assert not hasattr(user_config, "DroidConfig")


def test_library_import_does_not_install_logging_handler():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import mobilerun, logging; assert not logging.getLogger('mobilerun').handlers",
        ],
        check=True,
    )
