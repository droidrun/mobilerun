"""Location reported by a device."""

from pydantic import BaseModel


class Location(BaseModel):
    """Coordinates reported by the device's location service."""

    latitude: float
    longitude: float
