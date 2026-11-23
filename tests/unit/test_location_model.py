def test_location_is_a_shared_value_model() -> None:
    from mobilerun.models import Location
    from mobilerun.models.location import Location as LocationModel

    assert Location is LocationModel
    assert Location(latitude=52.52, longitude=13.405).model_dump() == {
        "latitude": 52.52,
        "longitude": 13.405,
    }
