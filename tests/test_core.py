import unittest

from xray_tools import Observation, energy_to_wavelength, wavelength_to_energy


class ObservationTests(unittest.TestCase):
    def test_exposure_is_converted_to_seconds(self) -> None:
        observation = Observation("Chandra", "ACIS-I", "12345", 42.5)

        self.assertEqual(observation.exposure_s, 42_500)

    def test_negative_exposure_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Observation("Chandra", "ACIS-I", "12345", -1)


class UnitConversionTests(unittest.TestCase):
    def test_energy_and_wavelength_round_trip(self) -> None:
        self.assertAlmostEqual(wavelength_to_energy(energy_to_wavelength(2.5)), 2.5)

    def test_non_positive_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            energy_to_wavelength(0)
        with self.assertRaises(ValueError):
            wavelength_to_energy(-1)
