"""Common models for X-ray observations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """Metadata common to observations from X-ray observatories.

    Parameters
    ----------
    observatory
        Mission or observatory name, such as ``"Chandra"`` or ``"XMM-Newton"``.
    instrument
        Instrument used for the observation.
    obs_id
        Observatory-specific observation identifier.
    exposure_ks
        Good-time exposure in kiloseconds.
    """

    observatory: str
    instrument: str
    obs_id: str
    exposure_ks: float

    def __post_init__(self) -> None:
        if self.exposure_ks < 0:
            raise ValueError("exposure_ks must be non-negative")

    @property
    def exposure_s(self) -> float:
        """Good-time exposure in seconds."""
        return self.exposure_ks * 1_000
