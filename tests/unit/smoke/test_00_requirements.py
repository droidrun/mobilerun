"""One diagnostic table, without disclosing credential values."""


def test_all_requirements(requirements):
    table = "\n".join(
        f"{name:28} {'ok' if ok else 'missing'}" for name, ok in requirements.items()
    )
    print("\n" + table)
    assert all(requirements.values()), "Smoke requirements:\n" + table
