from pathlib import Path


def test_cli_domains_have_readmes_and_no_common_package():
    cli = Path(__file__).resolve().parents[2] / "mobilerun" / "cli"
    assert not (cli / "common").exists()
    for domain in ("auth", "base", "devices", "phone_number", "mailbox", "proxy"):
        assert (cli / domain / "README.md").is_file(), domain
    assert not (cli / "comms.py").exists()
